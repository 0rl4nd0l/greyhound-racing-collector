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


def project_metadata(raw, paths):
    """Decode only explicitly named scalar paths; unknown payloads stay bytes.

    Paths use * for array positions. No scalar string is recursively decoded.
    Duplicate keys reject, including in opaque subtrees. This is a projection,
    not a claim that the original document contains no results.
    """
    allowed={tuple(path.split(".")) for path in paths}; i=0; absent=object()
    def space():
        nonlocal i
        while i<len(raw) and raw[i] in b" \t\r\n":i+=1
    def token_string():
        nonlocal i
        start=i;i+=1
        while i<len(raw):
            if raw[i]==92:i+=2
            elif raw[i]==34:i+=1;return raw[start:i]
            else:i+=1
        raise InputBoundary('JSON_FRAMING')
    def parse(path,depth=0):
        nonlocal i
        if depth>50:raise InputBoundary('JSON_DEPTH')
        space();kind=raw[i:i+1]
        if kind in (b'{',b'['):
            if path in allowed and not any(p[:len(path)]==path and len(p)>len(path) for p in allowed):raise InputBoundary('EXPECTED_METADATA_SCALAR:'+'.'.join(path))
            is_object=kind==b'{';closing=b'}' if is_object else b']';i+=1
            output={} if is_object else [];seen=set();space()
            if raw[i:i+1]==closing:i+=1;return output
            while True:
                space()
                if is_object:
                    if raw[i:i+1]!=b'"':raise InputBoundary('JSON_FRAMING')
                    key=json.loads(token_string())
                    if key in seen:raise InputBoundary('DUPLICATE_METADATA_KEY')
                    seen.add(key);space()
                    if raw[i:i+1]!=b':':raise InputBoundary('JSON_FRAMING')
                    i+=1
                else:key='*'
                child=parse(path+(key,),depth+1)
                if child is not absent and (not isinstance(child,(dict,list)) or child):
                    if is_object:output[key]=child
                    else:output.append(child)
                space();separator=raw[i:i+1];i+=1
                if separator==closing:return output
                if separator!=b',':raise InputBoundary('JSON_FRAMING')
        start=i
        if kind==b'"':token_string()
        else:
            while i<len(raw) and raw[i] not in b',]} \t\r\n':i+=1
            if start==i:raise InputBoundary('JSON_FRAMING')
        return json.loads(raw[start:i]) if path in allowed else absent
    result=parse(());space()
    if i!=len(raw) or not isinstance(result,dict):raise InputBoundary('JSON_FRAMING')
    return result


# Explicit pre-race inputs used by the unchanged feature generators. Unknown
# annotations and embedded payloads are never decoded or passed to replay.
PRE_RACE_SCALARS = """schema_version metadata_captured_at created_at capture_timestamp captured_at
race_url metadata_source_url metadata_is_leakage_safe target_distance target_grade
 target_distance_source target_grade_source target_metadata_source content_sha256 content_length
 track_condition weather weather_condition weather_track_metadata_source weather_track_metadata_source_url
 weather_track_metadata_is_leakage_safe target_grade_context_schema target_grade_equivalence_key
 target_grade_exact_value target_grade_race_date target_grade_race_number target_grade_race_url
 target_grade_source_url target_grade_source_sha256 target_grade_venue""".split()
FORM_PATHS = (PRE_RACE_SCALARS + ['race_info.'+key for key in
    'date venue race_number race_time url race_time_mapping_status race_time_source distance grade'.split()]
    + ['runner_completeness.runner_count']
    + ['prejump_shadow_metadata.'+key for key in PRE_RACE_SCALARS+['race_date','jump_time','status','source_url','target_distance_safe','target_grade_safe','distance','grade']])
FORM_PATHS += [prefix+'weather_track_metadata_source_url.'+source for prefix in ('','prejump_shadow_metadata.') for source in ('canonical_pre_race_page','sidecar_weather_track_metadata','explicit_csv_sidecar','open_meteo_forecast_api','sportsbet_pre_race_page')]
REQUEST_PATHS = ['race_id','retained_input_manifest_sha256'] + ['runners.*.'+k for k in ['box_number','display_name']]
RECEIPT_PATHS = ['captured_at'] + ['markets.win.*.'+k for k in ['box_number','dog_name','odds_decimal']]


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
        if not stamp and any(value.strip() for j,value in enumerate(fields) if j>=len(names) or names[j].strip() not in {'Dog Name','BOX','box_number'}):
            raise InputBoundary('UNDATED_NON_ROSTER_VALUE')
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
