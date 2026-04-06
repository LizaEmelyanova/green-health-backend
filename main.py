from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr, Field
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from PIL import Image
import io
import uuid
import time
from typing import Optional, List
import uvicorn
import traceback
from collections import defaultdict
from sqlalchemy import create_engine, Column, String, DateTime, Boolean, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.postgresql import UUID
from dotenv import load_dotenv
import os

import torch
from PIL import Image
from transformers import pipeline, AutoImageProcessor, AutoModelForImageClassification

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
SECRET_KEY = os.getenv("SECRET_KEY")

if not SECRET_KEY:
    raise ValueError("SECRET_KEY not set in .env file")

# база данных
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    pool_recycle=3600
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# модели
class User(Base):
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, index=True, nullable=False)
    name = Column(String(100), nullable=False)
    hashed_password = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)

class ActiveToken(Base):
    __tablename__ = "active_tokens"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    token = Column(String(500), unique=True, nullable=False, index=True)
    user_email = Column(String(255), index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)

class RefreshToken(Base):
    __tablename__ = "refresh_tokens"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    token = Column(String(500), unique=True, nullable=False, index=True)
    user_email = Column(String(255), index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# конфигурация
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

app = FastAPI(title="GREEN HEALTH API", version="1.0")
security = HTTPBearer()
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

# Логирование запросов
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Логирование всех HTTP запросов"""
    start_time = time.time()
    
    print(f"{request.method} {request.url.path}")
    print(f"Client: {request.client.host if request.client else 'unknown'}")
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    print(f"Status: {response.status_code} ({duration:.3f}s)")
    
    return response

# Проверка JWT
PUBLIC_PATHS = [
    "/api/register",
    "/api/login",
    "/api/refresh",
    "/docs",
]

@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """Автоматическая проверка JWT токена для защищенных маршрутов"""
    
    # Пропускаем публичные маршруты
    if request.url.path in PUBLIC_PATHS:
        return await call_next(request)
    
    # Проверяем наличие токена
    auth_header = request.headers.get("Authorization")
    
    if not auth_header:
        return JSONResponse(
            status_code=401,
            content={"detail": "Missing authorization token"},
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    if not auth_header.startswith("Bearer "):
        return JSONResponse(
            status_code=401,
            content={"detail": "Invalid authorization header format"},
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    token = auth_header.replace("Bearer ", "")
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        
        if payload.get("type") != "access":
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid token type"}
            )
        
        # Добавляем информацию о пользователе в request
        request.state.user_email = payload.get("sub")
        request.state.user_name = payload.get("name")
        
    except jwt.ExpiredSignatureError:
        return JSONResponse(
            status_code=401,
            content={"detail": "Token has expired"},
            headers={"WWW-Authenticate": "Bearer"}
        )
    except JWTError as e:
        return JSONResponse(
            status_code=401,
            content={"detail": f"Invalid token: {str(e)}"},
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    return await call_next(request)

# Ограничение скорости
request_counts = defaultdict(list)
RATE_LIMIT = 60  # запросов в минуту
RATE_WINDOW = 60

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Ограничение количества запросов от одного IP"""
    
    # Пропускаем публичные маршруты
    if request.url.path in ["/api/health", "/"]:
        return await call_next(request)
    
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    
    # Очищаем старые записи
    request_counts[client_ip] = [
        ts for ts in request_counts[client_ip]
        if now - ts < RATE_WINDOW
    ]
    
    # Проверяем лимит
    if len(request_counts[client_ip]) >= RATE_LIMIT:
        return JSONResponse(
            status_code=429,
            content={
                "detail": f"Too many requests. Limit: {RATE_LIMIT} requests per minute",
                "retry_after": int(RATE_WINDOW - (now - request_counts[client_ip][0]))
            }
        )
    
    request_counts[client_ip].append(now)
    return await call_next(request)

# Глобальная обработка ошибок
@app.middleware("http")
async def error_handling_middleware(request: Request, call_next):
    """Глобальная обработка ошибок"""
    try:
        return await call_next(request)
    except Exception as e:
        print(f"Unhandled error: {e}")
        print(traceback.format_exc())
        
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error",
                "path": request.url.path
            }
        )

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic модели
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


disease_treatments = {
    "Apple with Apple Scab": 
        "Удалите и уничтожьте пораженные листья и плоды. " \
        "Обработайте дерево фунгицидами (медьсодержащими препаратами). " \
        "Весной проведите профилактическое опрыскивание. " \
        "Обеспечьте хорошую циркуляцию воздуха путем обрезки.",
    "Apple with Black Rot":
        "Обрежьте пораженные ветви на 15-20 см ниже видимых симптомов. " \
        "Обработайте срезы садовым варом. " \
        "Опрыскайте дерево фунгицидами каждые 7-10 дней. " \
        "Удалите мумифицированные плоды. ",
    "Tomato with Late Blight":
        "Немедленно удалите и сожгите пораженные листья и плоды. " \
        "Обработайте растения медьсодержащими препаратами. " \
        "Прекратите полив на 3-5 дней. " \
        "Обеспечьте хорошую вентиляцию в теплице. ",
    "Tomato with Early Blight": 
        "Удалите нижние, пораженные листья. " \
        "Обработайте фунгицидами широкого спектра. " \
        "Улучшите циркуляцию воздуха. " \
        "Мульчируйте почву для предотвращения брызг. ",
    "Grape with Black Rot": 
        "Удалите пораженные грозди и листья. " \
        "Обработайте серосодержащими препаратами. " \
        "Обеспечьте хорошую вентиляцию куста. " \
        "Проведите обрезку для проветривания. ",
    "Strawberry with Leaf Scorch": 
        "Немедленно удалите и уничтожьте все пораженные листья (сожгите или выбросьте). " \
        "Обработайте растения фунгицидами: Топаз (2 мл на 10 л воды) или Хорус (3-4 г на 10 л воды). " \
        "Проведите 2-3 обработки с интервалом 7-10 дней. " \
        "После сбора урожая проведите повторную обработку. " \
        "Используйте биопрепараты: Фитоспорин-М (5 г на 10 л воды)."
}


# Функции
def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception:
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
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return None

# Получение текущего пользователя
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """Проверка токена и получение текущего пользователя"""
    token = credentials.credentials
    
    # Проверяем токен в БД
    active_token = db.query(ActiveToken).filter(ActiveToken.token == token).first()
    if not active_token:
        raise HTTPException(status_code=401, detail="Token expired or invalid")
    
    payload = decode_token(token)
    if not payload or payload.get("type") != "access":
        raise HTTPException(status_code=401, detail="Invalid token")
    
    email = payload.get("sub")
    user = db.query(User).filter(User.email == email, User.is_active == True).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    
    return user

# Эндпоинты
@app.post("/api/register", response_model=UserResponse, status_code=201)
async def register(user_data: UserRegister, db: Session = Depends(get_db)):
    try:
        existing = db.query(User).filter(User.email == user_data.email).first()
        if existing:
            raise HTTPException(400, "Email already registered")
        
        hashed = hash_password(user_data.password)
        new_user = User(
            id=uuid.uuid4(),
            email=user_data.email,
            name=user_data.name,
            hashed_password=hashed
        )
        
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        return UserResponse(
            id=str(new_user.id),
            email=new_user.email,
            name=new_user.name
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(500, f"Registration failed: {str(e)}")

@app.post("/api/login", response_model=TokenResponse)
async def login(user_data: UserLogin, db: Session = Depends(get_db)):
    try:
        user = db.query(User).filter(
            User.email == user_data.email, 
            User.is_active == True
        ).first()
        
        if not user or not verify_password(user_data.password, user.hashed_password):
            raise HTTPException(401, "Incorrect email or password")
        
        access_token = create_access_token(data={"sub": user.email, "name": user.name})
        refresh_token = create_refresh_token(user.email)
        
        # Удаляем старые токены
        db.query(ActiveToken).filter(ActiveToken.user_email == user.email).delete()
        db.query(RefreshToken).filter(RefreshToken.user_email == user.email).delete()
        
        # Сохраняем новые
        db.add(ActiveToken(
            token=access_token,
            user_email=user.email,
            expires_at=datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        ))
        db.add(RefreshToken(
            token=refresh_token,
            user_email=user.email,
            expires_at=datetime.utcnow() + timedelta(days=7)
        ))
        db.commit()
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(500, f"Login failed: {str(e)}")

@app.post("/api/refresh", response_model=TokenResponse)
async def refresh_token(request: RefreshTokenRequest, db: Session = Depends(get_db)):
    try:
        payload = decode_token(request.refresh_token)
        if not payload or payload.get("type") != "refresh":
            raise HTTPException(401, "Invalid refresh token")
        
        email = payload.get("sub")
        
        stored = db.query(RefreshToken).filter(
            RefreshToken.token == request.refresh_token,
            RefreshToken.user_email == email
        ).first()
        
        if not stored:
            raise HTTPException(401, "Invalid refresh token")
        
        user = db.query(User).filter(User.email == email, User.is_active == True).first()
        if not user:
            raise HTTPException(401, "User not found")
        
        access_token = create_access_token(data={"sub": email, "name": user.name})
        refresh_token = create_refresh_token(email)
        
        # Обновляем токены
        db.query(ActiveToken).filter(ActiveToken.user_email == email).delete()
        db.query(RefreshToken).filter(RefreshToken.user_email == email).delete()
        
        db.add(ActiveToken(
            token=access_token,
            user_email=email,
            expires_at=datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        ))
        db.add(RefreshToken(
            token=refresh_token,
            user_email=email,
            expires_at=datetime.utcnow() + timedelta(days=7)
        ))
        db.commit()
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(500, f"Token refresh failed: {str(e)}")
    
plant_processor = None
plant_model = None

@app.on_event("startup")
async def load_model():
    global plant_processor, plant_model
    print("Loading plant disease model...")
    plant_processor = AutoImageProcessor.from_pretrained(
        "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification",
        use_fast=True  # Используем быстрый процессор
    )
    plant_model = AutoModelForImageClassification.from_pretrained(
        "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
    )
    plant_model.eval()
    print("Model loaded successfully!")

def get_treatment_instructions(disease_name: str) -> str:
    """Получение инструкций по лечению на основе названия болезни"""
    
    # Поиск точного совпадения
    if disease_name in disease_treatments:
        return disease_treatments[disease_name]
    
    # Поиск частичного совпадения
    for key, description in disease_treatments.items():
        if disease_name.lower() in key.lower() or key.lower() in disease_name.lower():
            return description
    
    # Общие рекомендации, если болезнь не найдена
    return "Изолируйте пораженное растение. " \
            "Удалите видимые пораженные части. " \
            "Обратитесь к агроному для точной диагностики. " \
            "Обеспечьте растению оптимальные условия ухода."

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    """Диагностика растения"""
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(400, "File must be an image")

        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        inputs = plant_processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = plant_model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        predicted_class_id = probabilities.argmax().item()
        confidence = probabilities[0][predicted_class_id].item()
        
        id_as_int = int(predicted_class_id)
        
        # Пробуем получить метку
        if hasattr(plant_model.config, 'id2label'):
            # Проверяем наличие ключа как int
            if id_as_int in plant_model.config.id2label:
                disease_name = plant_model.config.id2label[id_as_int]
            # Проверяем как str
            elif str(id_as_int) in plant_model.config.id2label:
                disease_name = plant_model.config.id2label[str(id_as_int)]
            else:
                # Если не нашли, выводим информацию для отладки
                print(f"ID {id_as_int} не найден в метках")
                print(f"Доступные ID: {list(plant_model.config.id2label.keys())[:5]}...")
                disease_name = f"Unknown disease (class {id_as_int})"
        else:
            disease_name = f"Class {predicted_class_id}"
        
        confidence_percentage = f"{confidence * 100:.2f}%"

        treatment_info = ""
        if "healthy"  not in disease_name.lower():
            treatment_info = get_treatment_instructions(disease_name)
        
        return {
            "disease": disease_name,
            "confidence": confidence,
            "confidence_percentage": confidence_percentage,
            "timestamp": datetime.utcnow().isoformat(),
            "recommendations": treatment_info
        }
        
    except Exception as e:
        print(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Prediction failed: {str(e)}")

@app.post("/api/logout", response_model=LogoutResponse)
async def logout(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        token = credentials.credentials
        db.query(ActiveToken).filter(ActiveToken.token == token).delete()
        db.query(RefreshToken).filter(RefreshToken.user_email == current_user.email).delete()
        db.commit()
        
        return LogoutResponse(message="Successfully logged out", success=True)
        
    except Exception as e:
        db.rollback()
        raise HTTPException(500, f"Logout failed: {str(e)}")

@app.get("/api/me", response_model=UserResponse)
async def get_current_user_info(current_user = Depends(get_current_user)):
    return UserResponse(
        id=str(current_user.id),
        email=current_user.email,
        name=current_user.name
    )

@app.get("/api/health")
async def health_check(db: Session = Depends(get_db)):
    try:
        return {
            "status": "healthy",
            "database": "connected",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {"status": "unhealthy", "database": f"error: {str(e)}"}

@app.get("/")
async def root():
    return {
        "message": "Green Health API is running!",
        "version": "1.0",
        "security": "JWT + Middleware",
        "middleware": ["logging", "auth", "rate_limit", "security_headers", "error_handling"]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)