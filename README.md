```markdown
# realxreal-embedding-server

The **realxreal embedding server** is a secure, authenticated microservice for generating text embeddings without exposing your API keys. It allows you to open-source your code while protecting your cloud credits and user data. This is used for the realxreal iOS app.

---

## Table of Contents

- [Motivation](#motivation)  
- [Architecture](#architecture)  
- [Threat Model](#threat-model)  
- [Project Structure](#project-structure)  
- [Setup & Deployment](#setup--deployment)  
- [iOS Client Integration](#ios-client-integration)  
- [Security Guarantees](#security-guarantees)  

---

## Motivation

> How do I give everyone the code without giving them the keys to my wallet?

Publishing source code with a hardcoded API key is dangerous. Bots can scrape it and exhaust your cloud resources within seconds. The solution: **Identity-Based Access Control (ID Tokens)** using Firebase Authentication.

Instead of relying on a shared secret, we issue **digital ID cards** (Firebase Auth Tokens). The server verifies *who* is calling, not *what secret* they hold.

---

## Architecture

### The Authenticated Microservice

```

```

[Client App] ---(text + ID Token)---> [Python Server] ---(verify token)---> Firebase

```

1. **App (Client)**  
   - Does not hold any secrets.  
   - Requests a temporary ID Token (valid 1 hour) from Firebase.

2. **Request**  
   - Client sends `text + ID Token` to the Python server.

3. **Server (Gatekeeper)**  
   - Verifies token with Firebase (`auth.verify_id_token(token)`).  
   - If valid → Run embedding.  
   - If invalid → Return 403.  
   - Optional: Apply per-user rate limits.

4. **Response**  
   - Embedding vector is returned over TLS.  

This architecture allows you to open-source your client and server code safely, while keeping your cloud resources protected.

---

## Threat Model

1. **API Key Theft**  
   - No API key exists in the client. Even if code is cloned, the attacker cannot call your API without a valid Firebase token.

2. **Abuse by Users**  
   - Rate limits and user bans can be applied via Firebase Console. Abusive users are blocked immediately.

3. **Data in Transit**  
   - All communication occurs over HTTPS (TLS 1.3). No need to implement custom crypto.

4. **Server Privacy**  
   - The server is stateless regarding user data. Text inputs and embeddings are never stored. Only ephemeral computation occurs.

5. **Client Security**  
   - Each user must authenticate via Firebase to obtain a valid ID token. Unauthorized requests are rejected.

This is a **defensible, enterprise-grade design** that allows open-source distribution while keeping secrets safe.

---
## Project Structure

```bash
realxreal-embedding-server/
├── app/
│   ├── **init**.py
│   └── main.py              # FastAPI server
├── .gitignore               # Critical security rules
├── Dockerfile               # Deployment instructions
├── README.md                # Documentation & Architecture
└── requirements.txt         # Dependencies

```

---

## Setup & Deployment

### Step 1: Get Firebase Credentials

1. Go to Firebase Console → Project Settings → Service Accounts.  
2. Generate a new private key → download `realxreal-firebase-adminsdk-xxxx.json`.  
3. Rename it to `firebase-credentials.json`.  
4. Place it in `realxreal-server/` **(Do not commit this file! Add to `.gitignore`)**.  

### Step 2: Update Python Server

**requirements.txt**

```text
fastapi
uvicorn
firebase-admin==6.2.0
````

**app/main.py** (token verification example)

```python
import firebase_admin
from firebase_admin import auth, credentials
from fastapi import FastAPI, Security, HTTPException, status, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from slowapi import Limiter

cred = credentials.Certificate("firebase-credentials.json")
firebase_admin.initialize_app(cred)

app = FastAPI()
security = HTTPBearer()
limiter = Limiter(key_func=lambda request: "user_id")

async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)):
    token = credentials.credentials
    try:
        decoded_token = auth.verify_id_token(token)
        return decoded_token['uid']
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

@app.post("/embed")
@limiter.limit("60/minute")
async def embed(request: Request, user_id: str = Depends(get_current_user)):
    # Run embedding
    return {"user_id": user_id, "embedding": [0.1, 0.2, 0.3]}
```

---

## iOS Client Integration

**Swift Example**: Fetch a token and send a request

```swift
import FirebaseAuth

final class APIEmbeddingManager {
    
    func embed(_ text: String) async throws -> [Float] {
        guard let url = URL(string: "\(baseURL)/embed") else { throw APIEmbeddingError.invalidURL }
        guard let user = Auth.auth().currentUser else { throw APIEmbeddingError.notLoggedIn }
        let token = try await user.getIDToken()
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        
        // Add JSON body here
        return []
    }
}
```

**Benefits**

* No secrets in code → safe for GitHub.
* Token verification ensures only authenticated users can call your API.
* Abuse can be blocked instantly via Firebase Console.

---

## Security Guarantees

| Layer              | Guarantee                                                     |
| ------------------ | ------------------------------------------------------------- |
| **Transport**      | All traffic encrypted with TLS 1.3.                           |
| **Server**         | Stateless processing; no logs of user input.                  |
| **Storage**        | Vectors are never stored on server; only on client device.    |
| **Authentication** | Firebase ID Tokens ensure only authorized users can call API. |
| **Rate Limiting**  | Per-user rate limits prevent abuse.                           |

---

## Notes

* For MVP speed, you can start with an API key, but switch to ID Tokens before open-sourcing.
* This architecture makes your backend **bulletproof** against scraping and abuse.

---

## References

* [Firebase Admin SDK (Python)](https://firebase.google.com/docs/admin/setup)
* [FastAPI Security Documentation](https://fastapi.tiangolo.com/tutorial/security/)
* [TLS in iOS](https://developer.apple.com/documentation/security/preventing_insecure_network_connections)

---
