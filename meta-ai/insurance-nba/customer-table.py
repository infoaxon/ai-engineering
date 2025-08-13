from sqlalchemy import create_engine, Column, Integer, String, Float, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()


class Customer(Base):
    __tablename__ = "customers"

    customer_id = Column(Integer, primary_key=True)
    name = Column(String)
    age = Column(Integer)
    location = Column(String)
    last_login_days_ago = Column(Integer)
    active_policies = Column(JSON)
    policy_expiring_in_days = Column(Integer)
    missed_notifications = Column(Integer)
    email_engagement_score = Column(Float)


# Setup connection
engine = create_engine("sqlite:///customers.db")
SessionLocal = sessionmaker(bind=engine)
Base.metadata.create_all(bind=engine)
