"""Metadata-only checks before authorized historical-feature processing.

Never return an outcome, dog name, feature value or target prediction. A mixed
card/database is rejected whole before result-valued columns are decoded.
"""
from datetime import date
import csv
import json
import sqlite3


class InputBoundary(ValueError):
    """Only constant structural reason codes may leave the restricted process."""


def json_keys(raw):
    """Scan JSON framing; decode object keys only, never scalar string values."""
    i=0; keys=[]
    def whitespace():
        nonlocal i
        while i<len(raw) and raw[i] in b' \r\n\t': i+=1
    def string():
        nonlocal i
        start=i
        if raw[i:i+1]!=b'"': raise InputBoundary('JSON_FRAMING')
        i+=1
        while i<len(raw):
            if raw[i]==92: i+=2
            elif raw[i]==34:
                i+=1;return raw[start:i]
            else:i+=1
        raise InputBoundary('JSON_FRAMING')
    def value(depth=0):
        nonlocal i
        if depth>50: raise InputBoundary('JSON_DEPTH')
        whitespace();kind=raw[i:i+1]
        if kind==b'{':
            i+=1;whitespace()
            if raw[i:i+1]==b'}': i+=1;return
            while True:
                whitespace();keys.append(json.loads(string()));whitespace()
                if raw[i:i+1]!=b':': raise InputBoundary('JSON_FRAMING')
                i+=1;value(depth+1);whitespace()
                c=raw[i:i+1];i+=1
                if c==b'}': return
                if c!=b',': raise InputBoundary('JSON_FRAMING')
        elif kind==b'[':
            i+=1;whitespace()
            if raw[i:i+1]==b']':i+=1;return
            while True:
                value(depth+1);whitespace();c=raw[i:i+1];i+=1
                if c==b']':return
                if c!=b',':raise InputBoundary('JSON_FRAMING')
        elif kind==b'"':string()
        else:
            start=i
            while i<len(raw) and raw[i] not in b',]} \r\n\t':i+=1
            if start==i:raise InputBoundary('JSON_FRAMING')
    value();whitespace()
    if i!=len(raw):raise InputBoundary('JSON_FRAMING')
    return keys


def metadata_without_results(raw):
    forbidden={'result','results','winner','winner_name','winner_box','is_winner','finish_position','finishing_position','placings','target_result','plc'}
    if any(key.lower() in forbidden for key in json_keys(raw)):
        raise InputBoundary('RESULT_FIELD_IN_METADATA')
    return json.loads(raw)


def card_dates_before_decode(raw,target,captured):
    # The frozen parser's DATE projection never decodes other cell values. Also
    # reject rows with a result-shaped value but no dated historical identity.
    header,_,body=raw.partition(b'\n'); delimiter=b'|' if header.count(b'|')>header.count(b',') else b','
    names=next(csv.reader([header.decode('utf-8-sig')],delimiter=delimiter.decode()))
    if names.count('DATE')!=1:raise InputBoundary('HISTORY_DATE_COLUMN_MISSING_OR_AMBIGUOUS')
    date_column=names.index('DATE'); result_columns={i for i,n in enumerate(names) if n.strip().upper() in {'PLC','POS','POSITION','FINISH','FINISH_POSITION'}}
    if not result_columns:raise InputBoundary('HISTORY_RESULT_COLUMN_UNKNOWN')
    row=[];token=bytearray();quoted=False;i=0
    def check(fields):
        stamp=fields[date_column].strip() if len(fields)>date_column else b''
        has_result=any(j<len(fields) and fields[j].strip() for j in result_columns)
        if has_result and not stamp:raise InputBoundary('UNDATED_RESULT_ROW')
        if stamp:
            try:day=date.fromisoformat(stamp.decode('ascii'))
            except (ValueError,UnicodeError):raise InputBoundary('HISTORY_DATE_FORMAT') from None
            if day>=target or day>captured:raise InputBoundary('TARGET_SAME_DAY_OR_FUTURE_CARD_ROW')
    while i<len(body):
        c=body[i:i+1]
        if c==b'"':
            if quoted and body[i+1:i+2]==b'"':token.extend(b'"');i+=1
            else:quoted=not quoted
        elif not quoted and c in (delimiter,b'\n'):
            row.append(bytes(token));token.clear()
            if c==b'\n':check(row);row=[]
        else:token.extend(c)
        i+=1
    if quoted:raise InputBoundary('CSV_FRAMING')
    if token or row:row.append(bytes(token));check(row)
    from src.predictor.comparison_candidates import projected_dates
    return projected_dates(raw)  # frozen framing/date semantics must also pass


def database_dates_before_decode(path,target_race_id,target,captured):
    # No outcome-valued column is selected. Unattached rows reject the snapshot.
    with sqlite3.connect(path.as_uri()+'?mode=ro&immutable=1',uri=True) as conn:
        conn.execute('PRAGMA query_only=ON')
        def authorize(action,table,column,*unused):
            if action==sqlite3.SQLITE_READ and (table,column) not in {('race_metadata','race_id'),('race_metadata','race_date'),('dog_race_data','race_id')}:
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK
        conn.set_authorizer(authorize)
        rows=conn.execute('SELECT race_id,race_date FROM race_metadata').fetchall()
        dates=[];identities=set()
        for race_id,stamp in rows:
            if race_id in identities:raise InputBoundary('DUPLICATE_HISTORY_ID')
            identities.add(race_id)
            try:day=date.fromisoformat(str(stamp))
            except ValueError:raise InputBoundary('DATABASE_HISTORY_DATE_FORMAT') from None
            if race_id==target_race_id or day>=target or day>captured:raise InputBoundary('TARGET_SAME_DAY_OR_FUTURE_DB_ROW')
            dates.append(day)
        missing=conn.execute('SELECT count(*) FROM dog_race_data d LEFT JOIN race_metadata r ON d.race_id=r.race_id WHERE r.race_id IS NULL').fetchone()[0]
        if missing:raise InputBoundary('ORPHAN_HISTORY_ROWS')
    return dates
