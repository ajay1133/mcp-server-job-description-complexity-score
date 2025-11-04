"""
Training dataset for ML-based complexity scoring
Each entry contains:
- requirement: The text description
- complexity_score: Target score (Replit Agent 3 baseline = 100)
- estimated_hours: Time to complete (for skilled AI agent user)
- factors: List of complexity factors present
"""

TRAINING_DATA = [
    # Very Simple Tasks (Score: 10-30)
    {
        "requirement": "Create a simple HTML page with a title and paragraph",
        "complexity_score": 15,
        "estimated_hours": 0.5,
        "factors": ["basic_web"]
    },
    {
        "requirement": "Build a basic CSS styled landing page with header and footer",
        "complexity_score": 25,
        "estimated_hours": 1.2,
        "factors": ["basic_web"]
    },
    {
        "requirement": "Create a static webpage with HTML and CSS for a personal portfolio",
        "complexity_score": 28,
        "estimated_hours": 1.5,
        "factors": ["basic_web"]
    },
    {
        "requirement": "Make a simple contact form with HTML only",
        "complexity_score": 20,
        "estimated_hours": 0.8,
        "factors": ["basic_web"]
    },
    {
        "requirement": "Design a responsive landing page with CSS flexbox",
        "complexity_score": 32,
        "estimated_hours": 1.8,
        "factors": ["basic_web", "frontend"]
    },
    
    # Simple Tasks (Score: 30-60)
    {
        "requirement": "Build a simple REST API endpoint that returns JSON data",
        "complexity_score": 40,
        "estimated_hours": 2.5,
        "factors": ["backend", "api_integration"]
    },
    {
        "requirement": "Create a React component for displaying a list of items",
        "complexity_score": 45,
        "estimated_hours": 2.8,
        "factors": ["frontend"]
    },
    {
        "requirement": "Implement a basic SQLite database schema with two tables",
        "complexity_score": 38,
        "estimated_hours": 2.2,
        "factors": ["database"]
    },
    {
        "requirement": "Build a Flask API with three GET endpoints",
        "complexity_score": 48,
        "estimated_hours": 3.0,
        "factors": ["backend", "api_integration"]
    },
    {
        "requirement": "Create a simple Vue.js app with routing",
        "complexity_score": 52,
        "estimated_hours": 3.5,
        "factors": ["frontend"]
    },
    
    # Moderate Tasks (Score: 60-90)
    {
        "requirement": "Build a CRUD REST API with PostgreSQL database integration",
        "complexity_score": 70,
        "estimated_hours": 4.5,
        "factors": ["backend", "database", "api_integration"]
    },
    {
        "requirement": "Create a React dashboard with charts and data visualization",
        "complexity_score": 65,
        "estimated_hours": 4.2,
        "factors": ["frontend"]
    },
    {
        "requirement": "Implement user registration and login with password hashing",
        "complexity_score": 75,
        "estimated_hours": 5.0,
        "factors": ["backend", "security", "database"]
    },
    {
        "requirement": "Build a Django REST API with MySQL database and authentication",
        "complexity_score": 85,
        "estimated_hours": 6.0,
        "factors": ["backend", "database", "api_integration", "security"]
    },
    {
        "requirement": "Create a Node.js API with MongoDB and JWT authentication",
        "complexity_score": 82,
        "estimated_hours": 5.8,
        "factors": ["backend", "database", "security", "api_integration"]
    },
    {
        "requirement": "Develop a responsive web app with Next.js and API routes",
        "complexity_score": 78,
        "estimated_hours": 5.5,
        "factors": ["frontend", "backend", "api_integration"]
    },
    
    # Baseline Tasks (Score: 90-110) - Replit Agent 3 level
    {
        "requirement": "Build a full CRUD application with React frontend and Express backend",
        "complexity_score": 95,
        "estimated_hours": 7.2,
        "factors": ["frontend", "backend", "api_integration"]
    },
    {
        "requirement": "Create a blog platform with authentication, posts, and comments",
        "complexity_score": 100,
        "estimated_hours": 8.0,
        "factors": ["frontend", "backend", "database", "security"]
    },
    {
        "requirement": "Develop a REST API with OAuth integration and PostgreSQL database",
        "complexity_score": 105,
        "estimated_hours": 8.5,
        "factors": ["backend", "database", "api_integration", "security"]
    },
    {
        "requirement": "Build a task management app with React, Django, and user authentication",
        "complexity_score": 102,
        "estimated_hours": 8.2,
        "factors": ["frontend", "backend", "database", "security"]
    },
    {
        "requirement": "Create an e-commerce product catalog with shopping cart functionality",
        "complexity_score": 108,
        "estimated_hours": 8.8,
        "factors": ["frontend", "backend", "database", "api_integration"]
    },
    
    # Complex Tasks (Score: 110-140)
    {
        "requirement": "Build a full-stack e-commerce platform with payment integration using Stripe",
        "complexity_score": 125,
        "estimated_hours": 10.5,
        "factors": ["frontend", "backend", "database", "api_integration", "security"]
    },
    {
        "requirement": "Create a social media app with real-time messaging and notifications",
        "complexity_score": 135,
        "estimated_hours": 12.0,
        "factors": ["frontend", "backend", "database", "real_time", "security"]
    },
    {
        "requirement": "Develop a multi-tenant SaaS application with subscription management",
        "complexity_score": 130,
        "estimated_hours": 11.0,
        "factors": ["frontend", "backend", "database", "api_integration", "security", "scalability"]
    },
    {
        "requirement": "Build a real-time collaborative document editor with WebSocket support",
        "complexity_score": 128,
        "estimated_hours": 10.8,
        "factors": ["frontend", "backend", "database", "real_time", "scalability"]
    },
    {
        "requirement": "Create a video streaming platform with user authentication and content management",
        "complexity_score": 138,
        "estimated_hours": 12.5,
        "factors": ["frontend", "backend", "database", "security", "scalability"]
    },
    
    # Very Complex Tasks (Score: 140-170)
    {
        "requirement": "Build a microservices architecture with API gateway and service discovery",
        "complexity_score": 155,
        "estimated_hours": 15.0,
        "factors": ["backend", "api_integration", "deployment", "scalability"]
    },
    {
        "requirement": "Create an AI-powered recommendation system with machine learning integration",
        "complexity_score": 165,
        "estimated_hours": 17.0,
        "factors": ["backend", "ai_ml", "database", "scalability"]
    },
    {
        "requirement": "Develop a real-time analytics dashboard with Kafka and Redis caching",
        "complexity_score": 148,
        "estimated_hours": 14.0,
        "factors": ["frontend", "backend", "database", "real_time", "scalability"]
    },
    {
        "requirement": "Build a comprehensive testing suite with unit, integration, and E2E tests",
        "complexity_score": 142,
        "estimated_hours": 13.0,
        "factors": ["testing", "backend", "frontend"]
    },
    {
        "requirement": "Create a deployment pipeline with Docker, Kubernetes, and CI/CD automation",
        "complexity_score": 145,
        "estimated_hours": 13.5,
        "factors": ["deployment", "scalability"]
    },
    {
        "requirement": "Develop a multi-region distributed system with load balancing and failover",
        "complexity_score": 168,
        "estimated_hours": 17.5,
        "factors": ["backend", "deployment", "scalability", "database"]
    },
    
    # Expert Level Tasks (Score: 170+)
    {
        "requirement": "Build an enterprise-grade microservices platform with Kubernetes orchestration, service mesh, and observability",
        "complexity_score": 185,
        "estimated_hours": 20.0,
        "factors": ["backend", "deployment", "scalability", "testing"]
    },
    {
        "requirement": "Create a machine learning platform with model training, deployment, and monitoring infrastructure",
        "complexity_score": 195,
        "estimated_hours": 22.0,
        "factors": ["ai_ml", "backend", "deployment", "scalability", "database"]
    },
    {
        "requirement": "Develop a comprehensive security framework with encryption, PKI, and compliance monitoring",
        "complexity_score": 178,
        "estimated_hours": 19.0,
        "factors": ["security", "backend", "testing", "deployment"]
    },
    {
        "requirement": "Build a real-time multiplayer game server with WebSocket, state synchronization, and anti-cheat",
        "complexity_score": 192,
        "estimated_hours": 21.5,
        "factors": ["backend", "real_time", "database", "scalability", "security"]
    },
    {
        "requirement": "Create a blockchain-based application with smart contracts and distributed consensus",
        "complexity_score": 205,
        "estimated_hours": 24.0,
        "factors": ["backend", "security", "scalability", "database"]
    },
    {
        "requirement": "Design and implement a full DevOps platform with GitOps, automated testing, security scanning, and deployment orchestration",
        "complexity_score": 188,
        "estimated_hours": 20.5,
        "factors": ["deployment", "testing", "security", "scalability"]
    },
    
    # Additional diverse examples
    {
        "requirement": "Build a GraphQL API with Apollo Server and PostgreSQL",
        "complexity_score": 88,
        "estimated_hours": 6.5,
        "factors": ["backend", "api_integration", "database"]
    },
    {
        "requirement": "Create a mobile-responsive Progressive Web App with offline support",
        "complexity_score": 92,
        "estimated_hours": 7.0,
        "factors": ["frontend"]
    },
    {
        "requirement": "Implement serverless functions with AWS Lambda and DynamoDB",
        "complexity_score": 98,
        "estimated_hours": 7.8,
        "factors": ["backend", "database", "deployment"]
    },
    {
        "requirement": "Build a chatbot with natural language processing and OpenAI integration",
        "complexity_score": 115,
        "estimated_hours": 9.5,
        "factors": ["backend", "ai_ml", "api_integration"]
    },
    {
        "requirement": "Create a content management system with role-based access control",
        "complexity_score": 112,
        "estimated_hours": 9.0,
        "factors": ["frontend", "backend", "database", "security"]
    },
    {
        "requirement": "Develop a financial dashboard with real-time stock data and charting",
        "complexity_score": 118,
        "estimated_hours": 9.8,
        "factors": ["frontend", "backend", "real_time", "api_integration"]
    },
    {
        "requirement": "Build an IoT device management platform with MQTT protocol support",
        "complexity_score": 152,
        "estimated_hours": 14.5,
        "factors": ["backend", "real_time", "database", "scalability"]
    },
    {
        "requirement": "Create a video conferencing application with WebRTC",
        "complexity_score": 175,
        "estimated_hours": 18.5,
        "factors": ["frontend", "backend", "real_time", "scalability"]
    },
    {
        "requirement": "Implement a search engine with Elasticsearch and relevance ranking",
        "complexity_score": 145,
        "estimated_hours": 13.5,
        "factors": ["backend", "database", "scalability"]
    },
    {
        "requirement": "Build a monitoring and alerting system with Prometheus and Grafana",
        "complexity_score": 125,
        "estimated_hours": 10.5,
        "factors": ["backend", "deployment", "scalability"]
    },
]

# Test cases for validation
TEST_CASES = [
    {
        "requirement": "Create a simple HTML landing page with CSS styling",
        "expected_score_range": (20, 35),
        "expected_hours_range": (0.8, 2.0)
    },
    {
        "requirement": "Build a REST API with database and authentication",
        "expected_score_range": (80, 95),
        "expected_hours_range": (5.5, 7.5)
    },
    {
        "requirement": "Develop a full-stack application with React and Node.js",
        "expected_score_range": (95, 115),
        "expected_hours_range": (7.0, 10.0)
    },
    {
        "requirement": "Create a real-time chat application with WebSocket and MongoDB",
        "expected_score_range": (115, 140),
        "expected_hours_range": (9.5, 13.0)
    },
    {
        "requirement": "Build a microservices platform with Kubernetes and CI/CD",
        "expected_score_range": (150, 180),
        "expected_hours_range": (14.0, 20.0)
    },
]

def get_training_data():
    """Returns the training dataset"""
    return TRAINING_DATA

def get_test_cases():
    """Returns test cases for validation"""
    return TEST_CASES
