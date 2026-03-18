import os

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, create_engine, desc
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy.sql import func

import warnings
warnings.filterwarnings('ignore')

from dotenv import load_dotenv
load_dotenv()


# DATABASE CONNECTION
DB_HOST     = os.getenv('DB_HOST')
DB_PORT     = os.getenv('DB_PORT')
DB_NAME     = os.getenv('DB_NAME')
DB_USER     = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')

DATABASE_URL = (f"postgresql://{DB_USER}:{DB_PASSWORD}"
                f"@{DB_HOST}:{DB_PORT}/{DB_NAME}")

engine  = create_engine(DATABASE_URL, echo=False)
Session = sessionmaker(bind=engine)
Base    = declarative_base()


# Session start and end points
def get_session():
    """Get a new database session."""
    return Session()

def close_session(session):
    """Close a database session."""
    session.close()


# DATABASE MODELS (TABLES)
class Applicant(Base):
    """
    Applicant tracking table.
    One row per unique person identified by CNIC.
    Tracks full history across multiple visits.
    """
    __tablename__ = 'applicants'

    id                 = Column(Integer, primary_key=True, autoincrement=True)
    cnic               = Column(String(15), unique=True, nullable=False)
    first_seen         = Column(DateTime, default=func.now())
    last_seen          = Column(DateTime, default=func.now())

    # Visit tracking
    total_visits       = Column(Integer, default=0)
    total_approved     = Column(Integer, default=0)
    total_rejected     = Column(Integer, default=0)
    last_decision      = Column(String(10))

    # Last loan details
    last_loan_amnt     = Column(Float)
    last_interest_rate = Column(Float)

    # Relationship
    predictions = relationship('Prediction', back_populates='applicant',
                                cascade='all, delete-orphan')

    def __repr__(self):
        return (f"<Applicant(cnic={self.cnic}, "
                f"visits={self.total_visits}, "
                f"last_decision={self.last_decision})>")

class Prediction(Base):
    """
    Main predictions table.
    One row per API call — stores model outputs.
    """
    __tablename__ = 'predictions'

    id                      = Column(Integer, primary_key=True, autoincrement=True)
    timestamp               = Column(DateTime, default=func.now())
    applicant_id            = Column(Integer, ForeignKey('applicants.id'), nullable=False)

    # Classification
    decision                = Column(String(10))    # APPROVED / REJECTED
    default_probability     = Column(Float)
    threshold               = Column(Float)

    # Regression
    interest_rate           = Column(Float)         # None for existing loan

    # Clustering
    cluster_id              = Column(Integer)
    cluster_label           = Column(String(50))
    cluster_prob_high_value = Column(Float)
    cluster_prob_standard   = Column(Float)

    # Relationships
    applicant    = relationship('Applicant',   back_populates='predictions')
    inputs       = relationship('Input',       back_populates='prediction',
                                cascade='all, delete-orphan')
    explanations = relationship('Explanation', back_populates='prediction',
                                cascade='all, delete-orphan')

    def __repr__(self):
        return (f"<Prediction(id={self.id}, "
                f"decision={self.decision}, "
                f"prob={self.default_probability})>")

class Input(Base):
    """
    Raw input features table.
    One row per prediction — stores what the user submitted.
    """
    __tablename__ = 'inputs'

    id                        = Column(Integer, primary_key=True, autoincrement=True)
    prediction_id             = Column(Integer, ForeignKey('predictions.id'))
    timestamp                 = Column(DateTime, default=func.now())

    # Raw input features
    loan_amnt                 = Column(Float)
    loan_int_rate             = Column(Float)
    loan_grade                = Column(String(5))
    loan_percent_income       = Column(Float)
    loan_intent               = Column(String(50))
    person_income             = Column(Float)
    person_age                = Column(Integer)
    person_emp_length         = Column(Float)
    person_home_ownership     = Column(String(20))
    cb_person_default_on_file = Column(String(5))

    # Relationship
    prediction = relationship('Prediction', back_populates='inputs')

    def __repr__(self):
        return (f"<Input(id={self.id}, "
                f"prediction_id={self.prediction_id})>")

class Explanation(Base):
    """
    SHAP explanations table.
    Multiple rows per prediction — one row per SHAP feature.
    """
    __tablename__ = 'explanations'

    id            = Column(Integer, primary_key=True, autoincrement=True)
    prediction_id = Column(Integer, ForeignKey('predictions.id'))
    timestamp     = Column(DateTime, default=func.now())

    # Which model this explanation belongs to
    task          = Column(String(20))      # classification / regression

    # SHAP details
    feature_name  = Column(String(100))
    readable_name = Column(String(100))
    feature_value = Column(Float)
    shap_impact   = Column(Float)
    direction     = Column(String(50))      # increases / decreases risk

    # Relationship
    prediction = relationship('Prediction', back_populates='explanations')

    def __repr__(self):
        return (f"<Explanation(id={self.id}, "
                f"task={self.task}, "
                f"feature={self.feature_name})>")

class ModelMetadata(Base):
    """
    Model metadata table.
    Tracks which models are in production.
    """
    __tablename__ = 'model_metadata'

    id            = Column(Integer, primary_key=True, autoincrement=True)
    created_at    = Column(DateTime, default=func.now())
    task          = Column(String(20))
    model_name    = Column(String(100))
    model_version = Column(String(20))
    threshold     = Column(Float, nullable=True)

    def __repr__(self):
        return (f"<ModelMetadata(task={self.task}, "
                f"model={self.model_name})>")


# CREATE TABLES
def create_tables():
    """Create all tables in PostgreSQL if they don't exist."""
    Base.metadata.create_all(engine)
    print("All tables created successfully")



# CRUD — SAVE FUNCTIONS
def get_or_create_applicant(session, cnic, prediction_data, input_data):
    """
    Get existing applicant by CNIC or create new one.
    Updates visit stats on every call.
    - First visit  → creates new applicant row
    - Return visit → updates existing row stats
    Args:
        session        : SQLAlchemy session
        cnic           : applicant CNIC string
        prediction_data: dict with classification + regression keys
        input_data     : dict of raw user inputs
    Returns:
        Applicant: saved or updated applicant object
    """
    applicant = session.query(Applicant).filter(Applicant.cnic == cnic).first()

    decision  = prediction_data['classification']['decision']
    loan_amnt = input_data.get('loan_amnt')
    reg       = prediction_data.get('regression', {})
    rate      = reg.get('predicted_interest_rate') if reg else input_data.get('loan_int_rate')

    if applicant is None:
        applicant = Applicant(
            cnic               = cnic,
            total_visits       = 1,
            total_approved     = 1 if decision == 'APPROVED' else 0,
            total_rejected     = 1 if decision == 'REJECTED' else 0,
            last_decision      = decision,
            last_loan_amnt     = loan_amnt,
            last_interest_rate = rate
        )
        session.add(applicant)
    else:
        applicant.total_visits       += 1
        applicant.total_approved     += 1 if decision == 'APPROVED' else 0
        applicant.total_rejected     += 1 if decision == 'REJECTED' else 0
        applicant.last_decision       = decision
        applicant.last_loan_amnt      = loan_amnt
        applicant.last_interest_rate  = rate
        applicant.last_seen           = func.now()

    session.commit()
    session.refresh(applicant)
    return applicant


def save_prediction(session, prediction_data, applicant_id):
    cls = prediction_data['classification']
    reg = prediction_data.get('regression', {})
    clu = prediction_data['clustering']

    prediction = Prediction(
        applicant_id            = applicant_id,    # ← add this
        decision                = cls['decision'],
        default_probability     = cls['default_probability'],
        threshold               = cls['threshold'],
        interest_rate           = reg.get('predicted_interest_rate'),
        cluster_id              = clu['cluster_id'],
        cluster_label           = clu['cluster_label'],
        cluster_prob_high_value = clu['probabilities']['High Value Borrower'],
        cluster_prob_standard   = clu['probabilities']['Standard Borrower']
    )

    session.add(prediction)
    session.commit()
    session.refresh(prediction)
    return prediction


def save_input(session, prediction_id, input_data):
    """
    Save raw user inputs to inputs table.
    Args:
        session      : SQLAlchemy session
        prediction_id: id from saved prediction
        input_data   : dict of raw input features
    Returns:
        Input: saved input object
    """
    input_record = Input(
        prediction_id             = prediction_id,
        loan_amnt                 = input_data.get('loan_amnt'),
        loan_int_rate             = input_data.get('loan_int_rate'),
        loan_grade                = input_data.get('loan_grade'),
        loan_percent_income       = input_data.get('loan_percent_income'),
        loan_intent               = input_data.get('loan_intent'),
        person_income             = input_data.get('person_income'),
        person_age                = input_data.get('person_age'),
        person_emp_length         = input_data.get('person_emp_length'),
        person_home_ownership     = input_data.get('person_home_ownership'),
        cb_person_default_on_file = input_data.get('cb_person_default_on_file')
    )

    session.add(input_record)
    session.commit()
    session.refresh(input_record)
    return input_record


def save_explanation(session, prediction_id, explanations):
    """
    Save SHAP explanations to explanations table.
    Args:
        session      : SQLAlchemy session
        prediction_id: id from saved prediction
        explanations : list of explanation dicts
    Returns:
        list: saved explanation objects
    """
    saved = []
    for exp in explanations:
        explanation = Explanation(
            prediction_id = prediction_id,
            task          = exp.get('task'),
            feature_name  = exp.get('feature'),
            readable_name = exp.get('readable_name'),
            feature_value = exp.get('feature_value'),
            shap_impact   = exp.get('shap_impact'),
            direction     = exp.get('direction')
        )
        session.add(explanation)
        saved.append(explanation)

    session.commit()
    return saved


# CRUD — QUERY FUNCTIONS
def get_prediction_stats(session):
    """
    Get summary statistics of all predictions.
    Returns:
        dict: total, approved, rejected, averages
    """
    total    = session.query(Prediction).count()
    approved = session.query(Prediction).filter(
        Prediction.decision == 'APPROVED').count()
    rejected = session.query(Prediction).filter(
        Prediction.decision == 'REJECTED').count()
    avg_prob = session.query(
        func.avg(Prediction.default_probability)).scalar()
    avg_rate = session.query(
        func.avg(Prediction.interest_rate)).scalar()

    return {
        'total'                  : total,
        'approved'               : approved,
        'rejected'               : rejected,
        'avg_default_probability': round(float(avg_prob or 0), 4),
        'avg_interest_rate'      : round(float(avg_rate or 0), 4)
    }

def delete_applicant_by_cnic(session, cnic):
    """
    Delete applicant and ALL related records by CNIC.
    Cascade deletes:
        applicant → predictions → inputs + explanations
    Args:
        session: SQLAlchemy session
        cnic   : applicant CNIC string
    Returns:
        bool: True if deleted, False if not found
    """
    applicant = get_applicant_by_cnic(session, cnic)
    if applicant:
        session.delete(applicant)
        session.commit()
        return True
    return False

def get_applicant_by_cnic(session, cnic):
    """Get single applicant by CNIC."""
    return session.query(Applicant).filter(Applicant.cnic == cnic).first()

def get_applicant_history(session, cnic):
    """
    Get full prediction history for an applicant.
    Returns:
        tuple: (applicant, list of predictions)
    """
    applicant = get_applicant_by_cnic(session, cnic)
    if not applicant:
        return None, []

    predictions = session.query(Prediction).filter(Prediction.applicant_id == applicant.id).order_by(desc(Prediction.timestamp)).all()
    return applicant, predictions


# Final pipeline function
def save_full_prediction(session, cnic, prediction_data,
                          input_data, explanations):
    """
    Save complete prediction in a single function call.
    Handles applicant + prediction + input + explanations.
    Args:
        session        : SQLAlchemy session
        cnic           : applicant CNIC string
        prediction_data: dict from pipeline
        input_data     : dict of raw user inputs
        explanations   : list of SHAP explanation dicts
    Returns:
        dict: applicant, prediction, input, explanations
    """
    applicant  = get_or_create_applicant(session, cnic, prediction_data, input_data)
    prediction = save_prediction(session, prediction_data, applicant.id)

    input_record = save_input(session, prediction.id, input_data)
    exps         = save_explanation(session, prediction.id, explanations)

    return {
        'applicant'   : applicant,
        'prediction'  : prediction,
        'input'       : input_record,
        'explanations': exps
    }



# MAIN
if __name__ == "__main__":
    print("Testing database connection...")
    session = get_session()

    try:
        create_tables()
        stats = get_prediction_stats(session)
        print(f"\n Database connection successful")
        print(f"   Total predictions : {stats['total']}")
        print(f"   Approved          : {stats['approved']}")
        print(f"   Rejected          : {stats['rejected']}")

    except Exception as e:
        print(f" Database error: {e}")

    finally:
        close_session(session)