from fastapi import FastAPI
from pydantic import BaseModel, EmailStr

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from PIL import Image
import io
import uuid
from typing import Optional
import torch
import torchvision.transforms as transforms

# Конфигурация
SECRET_KEY = "your-secret-key-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Инициализация
app = FastAPI(title="GREEN HEALTH")
security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# База данных
users_db = {}           # email -> {id, email, name, hashed_password}
active_tokens = set()   # множество активных токенов

# Модели
class UserRegister(BaseModel):
    email: EmailStr
    password: str
    name: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: str
    email: EmailStr
    name: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

class PredictionResponse(BaseModel):
    disease: str
    confidence: float
    treatment: str
    description: Optional[str] = None

class LogoutResponse(BaseModel):
    message: str

# Дополнительные функции
def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str) -> Optional[dict]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Проверка токена и получение текущего пользователя"""
    token = credentials.credentials
    
    # Проверяем, не вышел ли пользователь
    if token not in active_tokens:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired or invalid"
        )
    
    payload = decode_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    
    email = payload.get("sub")
    if not email or email not in users_db:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    return users_db[email]

model = ""  # импорт модели


# Эндпоинты
@app.post("/register", response_model=UserResponse)
async def register(user_data: UserRegister):
    """Регистрация нового пользователя"""
    # Проверяем, не занят ли email
    if user_data.email in users_db:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Создаем пользователя
    user_id = str(uuid.uuid4())
    users_db[user_data.email] = {
        "id": user_id,
        "email": user_data.email,
        "name": user_data.name,
        "hashed_password": hash_password(user_data.password)
    }
    
    return UserResponse(
        id=user_id,
        email=user_data.email,
        name=user_data.name
    )

@app.post("/login", response_model=TokenResponse)
async def login(user_data: UserLogin):
    """Вход в систему, получение JWT токена"""
    # Ищем пользователя
    user = users_db.get(user_data.email)
    if not user or not verify_password(user_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )
    
    # Создаем токен
    access_token = create_access_token(
        data={"sub": user["email"], "name": user["name"]}
    )
    
    # Сохраняем токен в активные
    active_tokens.add(access_token)
    
    return TokenResponse(access_token=access_token)

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    current_user = Depends(get_current_user)
):
    """Загрузка фото растения и получение диагноза"""
    # Проверка типа файла
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )
    
    try:
        # Читаем и открываем изображение
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Получаем предсказание от модели
        result = model.predict(image)
        
        # Добавляем имя пользователя в лог (опционально)
        print(f"User {current_user['email']} uploaded {file.filename}")
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process image: {str(e)}"
        )

@app.post("/logout", response_model=LogoutResponse)
async def logout(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    current_user = Depends(get_current_user)
):
    """Выход из системы (делает токен недействительным)"""
    token = credentials.credentials
    
    # Удаляем токен из активных
    if token in active_tokens:
        active_tokens.remove(token)
    
    return LogoutResponse(message="Successfully logged out")

@app.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user = Depends(get_current_user)):
    """
    Получение информации о текущем пользователе (полезно для проверки авторизации)
    """
    return UserResponse(
        id=current_user["id"],
        email=current_user["email"],
        name=current_user["name"]
    )

@app.get("/")
async def root():
    return {
        "message": "Green Health API is running!",
        "version": "1.0 (simplified)",
        "endpoints": {
            "register": "POST /register",
            "login": "POST /login", 
            "predict": "POST /predict (auth required)",
            "logout": "POST /logout (auth required)",
            "me": "GET /me (auth required)"
        }
    }

# Запуск
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)