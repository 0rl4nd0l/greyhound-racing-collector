"""
SQLAlchemy models for the greyhound racing database.
This file defines the database schema for Alembic migrations and consistency tests.
"""

from sqlalchemy import (Column, Integer, String, Float, Date, Text, Boolean, 
                       DateTime, ForeignKey, Index, UniqueConstraint, Numeric)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()


class RaceMetadata(Base):
    __tablename__ = 'race_metadata'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    race_id = Column(String, unique=True)
    venue = Column(String)
    race_number = Column(Integer)
    race_date = Column(Date)
    race_name = Column(String)
    grade = Column(String)
    distance = Column(String)
    track_condition = Column(String)
    weather = Column(String)
    temperature = Column(Float)
    humidity = Column(Float)
    wind_speed = Column(Float)
    wind_direction = Column(String)
    track_record = Column(String)
    prize_money_total = Column(Float)
    prize_money_breakdown = Column(String)
    race_time = Column(String)
    field_size = Column(Integer)
    url = Column(String)
    extraction_timestamp = Column(DateTime)
    data_source = Column(String)
    winner_name = Column(String)
    winner_odds = Column(Float)
    winner_margin = Column(Float)
    race_status = Column(String)
    data_quality_note = Column(String)
    actual_field_size = Column(Integer)
    scratched_count = Column(Integer)
    scratch_rate = Column(Float)
    box_analysis = Column(String)
    weather_condition = Column(String)
    precipitation = Column(Float)
    pressure = Column(Float)
    visibility = Column(Float)
    weather_location = Column(String)
    weather_timestamp = Column(DateTime)
    weather_adjustment_factor = Column(Float)
    sportsbet_url = Column(String)
    venue_slug = Column(String)
    start_datetime = Column(DateTime)

    # Indexes for performance
    __table_args__ = (
        Index('idx_race_metadata_venue_date', 'venue', 'race_date'),
        Index('idx_race_metadata_race_id', 'race_id'),
        UniqueConstraint('race_id', name='uq_race_metadata_race_id'),
    )


class DogRaceData(Base):
    __tablename__ = 'dog_race_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    race_id = Column(String, ForeignKey('race_metadata.race_id', ondelete='CASCADE'))
    dog_name = Column(String)
    dog_clean_name = Column(String)
    dog_id = Column(Integer)
    box_number = Column(Integer)
    trainer_name = Column(String)
    trainer_id = Column(Integer)
    weight = Column(Float)
    running_style = Column(String)
    odds_decimal = Column(Float)
    odds_fractional = Column(String)
    starting_price = Column(Float)
    individual_time = Column(String)
    sectional_1st = Column(String)
    sectional_2nd = Column(String)
    sectional_3rd = Column(String)
    margin = Column(String)
    beaten_margin = Column(Float)
    was_scratched = Column(Boolean)
    blackbook_link = Column(String)
    extraction_timestamp = Column(DateTime)
    data_source = Column(String)
    form_guide_json = Column(Text)
    historical_records = Column(Text)
    performance_rating = Column(Float)
    speed_rating = Column(Float)
    class_rating = Column(Float)
    recent_form = Column(String)
    win_probability = Column(Float)
    place_probability = Column(Float)
    scraped_trainer_name = Column(String)
    scraped_reaction_time = Column(String)
    scraped_nbtt = Column(String)
    scraped_race_classification = Column(String)
    scraped_raw_result = Column(String)
    scraped_finish_position = Column(String)
    best_time = Column(Float)
    data_quality_note = Column(String)
    finish_position = Column(Integer)
    odds = Column(String)
    trainer = Column(String)
    winning_time = Column(String)
    placing = Column(Integer)
    form = Column(String)

    # Indexes for performance and foreign key relationships
    __table_args__ = (
        Index('idx_dog_race_data_race_id', 'race_id'),
        Index('idx_dog_race_data_dog_name', 'dog_clean_name'),
        Index('idx_dog_race_data_finish_position', 'finish_position'),
        Index('idx_dog_name', 'dog_clean_name'),
        UniqueConstraint('race_id', 'dog_clean_name', 'box_number', 
                        name='idx_dog_race_unique'),
    )


class Dogs(Base):
    __tablename__ = 'dogs'
    
    dog_id = Column(Integer, primary_key=True, autoincrement=True)
    dog_name = Column(String, unique=True, nullable=False)
    total_races = Column(Integer, default=0)
    total_wins = Column(Integer, default=0)
    total_places = Column(Integer, default=0)
    best_time = Column(Float)
    average_position = Column(Float)
    last_race_date = Column(String)
    created_at = Column(DateTime, default=func.current_timestamp())
    weight = Column(Numeric(5, 2))
    age = Column(Integer)
    id = Column(Integer)
    color = Column(String)
    owner = Column(String)
    trainer = Column(String)
    sex = Column(String)

    # Indexes for performance
    __table_args__ = (
        Index('idx_dogs_clean_name', 'dog_name'),
        Index('idx_dogs_trainer', 'trainer'),
    )


# Additional tables for completeness (if they exist in your database)
class MLModelRegistry(Base):
    __tablename__ = 'ml_model_registry'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String, nullable=False)
    model_version = Column(String, nullable=False)
    model_type = Column(String)
    file_path = Column(String)
    metrics = Column(Text)
    parameters = Column(Text)
    training_data_hash = Column(String)
    is_active = Column(Boolean, default=False)
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp())

    __table_args__ = (
        UniqueConstraint('model_name', 'model_version', name='uq_model_registry_name_version'),
    )


class PredictionHistory(Base):
    __tablename__ = 'prediction_history'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    race_id = Column(String, ForeignKey('race_metadata.race_id'))
    model_name = Column(String)
    model_version = Column(String)
    prediction_data = Column(Text)
    confidence_score = Column(Float)
    actual_results = Column(Text)
    accuracy_score = Column(Float)
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp())

    __table_args__ = (
        Index('idx_prediction_history_race_id', 'race_id'),
        Index('idx_prediction_history_model', 'model_name', 'model_version'),
    )
