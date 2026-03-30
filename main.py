from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from PIL import Image
import io
import uuid
from typing import Optional, Dict, Set
import uvicorn
import traceback
from sqlalchemy import create_engine, Column, String, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.postgresql import UUID
from dotenv import load_dotenv
import os

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
SECRET_KEY = os.getenv("SECRET_KEY", "default-secret-key")

# Создаем движок SQLAlchemy
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    """Модель пользователя в БД"""
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, index=True, nullable=False)
    name = Column(String(100), nullable=False)
    hashed_password = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)

class ActiveToken(Base):
    """Модель активных токенов"""
    __tablename__ = "active_tokens"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    token = Column(String(500), unique=True, nullable=False, index=True)
    user_email = Column(String(255), index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)

class RefreshToken(Base):
    """Модель refresh токенов"""
    __tablename__ = "refresh_tokens"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    token = Column(String(500), unique=True, nullable=False, index=True)
    user_email = Column(String(255), index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)

Base.metadata.create_all(bind=engine)

def get_db():
    """Получение сессии базы данных"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

SECRET_KEY = "your-secret-key-change-in-production-2024"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

app = FastAPI(title="GREEN HEALTH API", version="1.0")
security = HTTPBearer()

# Настройка хэширования паролей
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

class UserRegister(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6)
    name: str = Field(..., min_length=2)

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: str
    email: EmailStr
    name: str

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int

class LogoutResponse(BaseModel):
    message: str
    success: bool

class RefreshTokenRequest(BaseModel):
    refresh_token: str

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception as e:
        print(f"Password verification error: {e}")
        return False

def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire, "type": "access"})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def create_refresh_token(email: str) -> str:
    expire = datetime.utcnow() + timedelta(days=7)
    to_encode = {"sub": email, "exp": expire, "type": "refresh"}
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str) -> Optional[dict]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError as e:
        print(f"Token decode error: {e}")
        return None

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """Проверка токена и получение текущего пользователя из БД"""
    token = credentials.credentials
    
    # Проверяем, есть ли токен в активных
    active_token = db.query(ActiveToken).filter(ActiveToken.token == token).first()
    if not active_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired or invalid",
        )
    
    payload = decode_token(token)
    if not payload or payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )
    
    email = payload.get("sub")
    user = db.query(User).filter(User.email == email, User.is_active == True).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )
    
    return user

@app.post("/api/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserRegister, db: Session = Depends(get_db)):
    """Регистрация нового пользователя"""
    try:
        print(f"Registration attempt for: {user_data.email}")
        
        # Проверяем, не занят ли email
        existing_user = db.query(User).filter(User.email == user_data.email).first()
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Создаем пользователя
        hashed_password = hash_password(user_data.password)
        new_user = User(
            id=uuid.uuid4(),
            email=user_data.email,
            name=user_data.name,
            hashed_password=hashed_password
        )
        
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        print(f"User registered successfully: {user_data.email}")
        
        return UserResponse(
            id=str(new_user.id),
            email=new_user.email,
            name=new_user.name
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Registration error: {traceback.format_exc()}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {str(e)}"
        )

@app.post("/api/login", response_model=TokenResponse)
async def login(user_data: UserLogin, db: Session = Depends(get_db)):
    """Вход в систему, получение JWT токенов"""
    try:
        print(f"Login attempt for: {user_data.email}")
        
        user = db.query(User).filter(User.email == user_data.email, User.is_active == True).first()
        if not user or not verify_password(user_data.password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password"
            )
        
        access_token = create_access_token(data={"sub": user.email, "name": user.name})
        refresh_token = create_refresh_token(user.email)
        
        # Сохраняем access токен
        active_token = ActiveToken(
            token=access_token,
            user_email=user.email,
            expires_at=datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        )
        db.add(active_token)
        
        # Сохраняем refresh токен (удаляем старый, если был)
        db.query(RefreshToken).filter(RefreshToken.user_email == user.email).delete()
        refresh_token_obj = RefreshToken(
            token=refresh_token,
            user_email=user.email,
            expires_at=datetime.utcnow() + timedelta(days=7)
        )
        db.add(refresh_token_obj)
        db.commit()
        
        print(f"User logged in successfully: {user_data.email}")
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Login error: {traceback.format_exc()}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Login failed: {str(e)}"
        )

@app.post("/api/refresh", response_model=TokenResponse)
async def refresh_token(request: RefreshTokenRequest, db: Session = Depends(get_db)):
    """Обновление access токена по refresh токену"""
    try:
        payload = decode_token(request.refresh_token)
        if not payload or payload.get("type") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
        
        email = payload.get("sub")
        
        # Проверяем refresh токен в БД
        stored_refresh = db.query(RefreshToken).filter(
            RefreshToken.token == request.refresh_token,
            RefreshToken.user_email == email
        ).first()
        
        if not stored_refresh:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
        
        user = db.query(User).filter(User.email == email, User.is_active == True).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )
        
        # Создаем новые токены
        access_token = create_access_token(data={"sub": email, "name": user.name})
        refresh_token = create_refresh_token(email)
        
        # Удаляем старые токены
        db.query(ActiveToken).filter(ActiveToken.user_email == email).delete()
        db.query(RefreshToken).filter(RefreshToken.user_email == email).delete()
        
        # Сохраняем новые
        new_active = ActiveToken(
            token=access_token,
            user_email=email,
            expires_at=datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        )
        new_refresh = RefreshToken(
            token=refresh_token,
            user_email=email,
            expires_at=datetime.utcnow() + timedelta(days=7)
        )
        db.add_all([new_active, new_refresh])
        db.commit()
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Refresh error: {traceback.format_exc()}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Token refresh failed: {str(e)}"
        )

@app.post("/api/logout", response_model=LogoutResponse)
async def logout(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Выход из системы"""
    try:
        token = credentials.credentials
        
        # Удаляем токены из БД
        db.query(ActiveToken).filter(ActiveToken.token == token).delete()
        db.query(RefreshToken).filter(RefreshToken.user_email == current_user.email).delete()
        db.commit()
        
        return LogoutResponse(
            message="Successfully logged out",
            success=True
        )
        
    except Exception as e:
        print(f"Logout error: {traceback.format_exc()}")
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Logout failed: {str(e)}"
        )

@app.get("/api/me", response_model=UserResponse)
async def get_current_user_info(current_user = Depends(get_current_user)):
    """Получение информации о текущем пользователе"""
    return UserResponse(
        id=str(current_user.id),
        email=current_user.email,
        name=current_user.name
    )

@app.get("/api/health")
async def health_check(db: Session = Depends(get_db)):
    """Проверка состояния API"""
    try:
        # Проверяем подключение к БД
        user_count = db.query(User).count()
        return {
            "status": "healthy",
            "database": "connected",
            "users_count": user_count,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "database": f"error: {str(e)}",
            "timestamp": datetime.utcnow().isoformat()
        }

@app.get("/")
async def root():
    return {
        "message": "Green Health API is running!",
        "version": "1.0",
        "database": "PostgreSQL",
        "frontend_url": "http://localhost:5173",
        "hash_method": "pbkdf2_sha256"
    }

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )