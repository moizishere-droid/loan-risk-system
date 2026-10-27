"""Persistent accounts with salted password hashes and revocable bearer sessions."""
import hashlib
import hmac
import re
import secrets
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from database.database import AuthSession, User, get_session

router = APIRouter(prefix='/auth', tags=['Accounts'])
bearer = HTTPBearer(auto_error=False)


def get_db():
    db = get_session()
    try:
        yield db
    finally:
        db.close()


class Credentials(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=8, max_length=128)

    @field_validator('email')
    @classmethod
    def valid_email(cls, value):
        value = value.strip().casefold()
        if not re.fullmatch(r'[^\s@]+@[^\s@]+\.[^\s@]+', value):
            raise ValueError('Enter a valid email address')
        return value


class Registration(Credentials):
    full_name: str = Field(min_length=1, max_length=120)

    @field_validator('full_name')
    @classmethod
    def valid_name(cls, value):
        value = value.strip()
        if not value:
            raise ValueError('Enter your name')
        return value


def password_hash(password, salt=None):
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac('sha256', password.encode(), bytes.fromhex(salt), 600_000)
    return f'{salt}${digest.hex()}'


def current_user(credentials: HTTPAuthorizationCredentials = Depends(bearer),
                 db: Session = Depends(get_db)):
    if credentials is None:
        raise HTTPException(401, 'Please log in to continue.', headers={'WWW-Authenticate': 'Bearer'})
    token_hash = hashlib.sha256(credentials.credentials.encode()).hexdigest()
    session = db.get(AuthSession, token_hash)
    if session is None or session.expires_at <= datetime.utcnow():
        raise HTTPException(401, 'Your session has expired. Please log in again.')
    user = db.get(User, session.user_id)
    if user is None:
        raise HTTPException(401, 'Please log in to continue.')
    return user


def user_info(user):
    return {'id': user.id, 'full_name': user.full_name, 'email': user.email}


def new_session(db, user):
    # Remove expired tokens rather than accumulating them indefinitely.
    db.query(AuthSession).filter(AuthSession.expires_at <= datetime.utcnow()).delete()
    token = secrets.token_urlsafe(32)
    db.add(AuthSession(token_hash=hashlib.sha256(token.encode()).hexdigest(), user_id=user.id,
                       expires_at=datetime.utcnow() + timedelta(hours=24)))
    db.commit()
    return {'token': token, 'user': user_info(user)}


@router.post('/register', status_code=201)
def register(data: Registration, db: Session = Depends(get_db)):
    user = User(email=data.email, full_name=data.full_name, password_hash=password_hash(data.password))
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, 'An account with this email already exists. Please log in.')
    db.refresh(user)
    return new_session(db, user)


@router.post('/login')
def login(data: Credentials, db: Session = Depends(get_db)):
    user = db.query(User).filter_by(email=data.email).first()
    stored = user.password_hash if user else password_hash('dummy-password')
    matches = hmac.compare_digest(stored, password_hash(data.password, stored.split('$')[0]))
    if user is None or not matches:
        raise HTTPException(401, 'Incorrect email or password.')
    return new_session(db, user)


@router.get('/me')
def me(user: User = Depends(current_user)):
    return user_info(user)


@router.post('/logout')
def logout(user: User = Depends(current_user),
           credentials: HTTPAuthorizationCredentials = Depends(bearer),
           db: Session = Depends(get_db)):
    session = db.get(AuthSession, hashlib.sha256(credentials.credentials.encode()).hexdigest())
    if session:
        db.delete(session)
        db.commit()
    return {'success': True}
