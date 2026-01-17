# StressEase Backend API

A Flask-based REST API for mental health support, featuring AI-powered chat assistance, mood tracking, and crisis intervention. Built with Firebase (Authentication + Firestore) and Google Gemini dual-model architecture.

## 🌟 Features

- **AI Chat Support** - Empathetic conversational AI using Google Gemini with crisis detection
- **Mood Tracking** - Daily 12-question mood quiz with DASS-21 integration
- **Crisis Resources** - Country-specific emergency helplines and mental health resources
- **SOS Emergency Contacts** - Real-time emergency contacts with agentic architecture (SerpApi + LLM)
- **Analytics & Predictions** - 7-day trend analysis with hybrid prediction (rule-based + LLM explanations)
- **Stress Prediction** - LLM-powered stress forecasting based on mood, chat, and quiz data
- **Session Management** - Persistent chat sessions with conversation history
- **Dual-Model LLM** - Optimized AI architecture for cost and performance
- **Firebase Integration** - Secure authentication and real-time data storage

## 🏗️ Architecture

### Dual-Model LLM Architecture

**Base Model** (gemini-2.0-flash-lite):

- Mood log summarization
- AI insights generation
- Crisis resource generation
- Temperature: 0.3 (factual, deterministic)

**Advanced Model** (gemini-2.0-flash-lite → gemini-2.0-flash in production):

- Chat responses and conversations
- Context-aware emotional support
- Temperature: 0.7 (creative, empathetic)

### SOS Tool-Based Pipeline Architecture
 
 The SOS emergency contacts feature uses a robust **multi-stage linear pipeline** for reliability and speed:
 
 1. **Smart Caching Layer**: Checks Firestore cache with 30-day TTL.
 2. **Web Search Stage**: Uses SerpApi to deterministically execute 3 targeted search queries when cache is stale.
 3. **LLM Extraction Stage**: Base model (gemini-2.0-flash-lite) processes the raw search results to extract and structure contact data:
    - Exactly 5 contacts (1 national emergency + 4 mental health crisis hotlines)
    - Validates official sources (.gov, .org domains)
    - Ensures information is current for the present year.
 4. **Validation Stage**: Verifies data structure, contact counts, and required fields before accepting the result.
 5. **Fallback Mechanism**: Returns cached data if fresh fetch fails to ensure user safety.
 
 This **deterministic tool-based pipeline** ensures users always get reliable, up-to-date emergency contacts with 100% consistency, avoiding the unpredictability of autonomous agents in critical safety scenarios.

### Technology Stack

- **Framework**: Flask 2.3.3
- **AI/ML**: LangChain 1.1.0, Google Gemini API
- **Web Search**: SerpApi for real-time emergency contact data
- **Database**: Firebase Firestore
- **Auth**: Firebase Authentication (handled by Android app)
- **Language**: Python 3.8+

## 📋 Requirements

- Python 3.8 or higher
- Firebase project with Firestore enabled
- Google Gemini API key
- Firebase service account credentials

## 🚀 Setup

### 1. Clone and Navigate

```bash
git clone <repository-url>
cd StressEase
```

### 2. Create Virtual Environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux:**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
# Google Gemini API
GEMINI_API_KEY=your_gemini_api_key_here

# Firebase Configuration
FIREBASE_CREDENTIALS_PATH=firebase-credentials.json

# Flask Configuration
SECRET_KEY=your-secret-key-change-in-production
FLASK_DEBUG=True
```

### 5. Add Firebase Credentials

Download your Firebase service account JSON from:

- Firebase Console → Project Settings → Service Accounts → Generate New Private Key

Save it as `firebase-credentials.json` in the project root.

### 6. Run the Application

```bash
python run.py
```

The server will start at: **<http://localhost:5000>**

## 📡 API Endpoints

**Base URL:** `http://localhost:5000`

**Authentication:** All endpoints (except `/health`) require Firebase ID token in header:

```text
Authorization: Bearer <firebase_id_token>
```

---

### Health Check

- **GET** `/health` - Returns server status

### Mood Tracking

- **POST** `/api/mood/quiz/daily` - Submit daily 12-question mood quiz with core scores, rotating domain scores, and DASS-21 metrics. Automatically computes averages, identifies high/low points, and triggers weekly DASS aggregation after every 7 submissions.

### Chat Support

- **POST** `/api/chat/message` - Send chat message. Creates new session if `session_id` is null. Returns AI response with crisis detection, personalized context, and conversation history (last 25 messages).
- **POST** `/api/chat/end-session` - End chat session and cleanup server resources.
- **GET/POST** `/api/chat/crisis-resources?country=<country>` - Get country-specific emergency services, crisis hotlines, and mental health resources. Uses agentic architecture with SerpApi + LLM for real-time data when cache is stale (30-day TTL). Returns exactly 5 contacts with graceful fallback.

### Stress Prediction

- **POST** `/api/predict` - Predict tomorrow's stress level using 7-day metrics. Calculates backend-verified `avgQuizScore` from Firestore, accepts `avgMoodScore` (1.0-5.0) and `chatCount` (0-999). Returns stress probability, label (Low/Medium/High), confidence score, and LLM-generated explanation.

### Analytics

- **POST** `/api/analytics/final-summary` - Generate comprehensive 7-day analytics summary. Backend fetches mood logs internally. Returns:
  - **Summary**: `avg_mood`, `avg_stress`, `dominant_issue` (depression/anxiety/stress)
  - **Trends**: Mood and stress trends (increasing/declining/stable)
  - **Prediction**: Hybrid prediction with rule-based state determination and LLM-generated human-relatable explanation
  - **Metadata**: Data quality assessment and recommendations

---

## 🔒 Security

**Authentication:** Android app obtains Firebase ID token → Backend verifies with Firebase Admin SDK → User ID extracted for database operations.

**Protection:** All endpoints use `@token_required` decorator. Users can only access their own data. Configure Firestore security rules to restrict access by user ID.

## 🧪 Testing

```bash
python test_backend.py
```

## 🤝 Android App Integration

The Android app handles:

- User registration and login (Firebase Auth)
- User profile management (Firestore)
- Emergency contacts (Firestore)
- Chat session listing and deletion (Firestore)
- Real-time updates (Firestore listeners)

The backend handles:

- Mood quiz processing and storage
- AI chat responses with personalization
- Crisis resource generation
- Session-based conversation history.
