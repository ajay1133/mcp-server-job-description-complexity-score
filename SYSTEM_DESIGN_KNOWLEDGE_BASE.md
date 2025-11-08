# System Design Knowledge Base for AI Model Training

## Purpose

This document serves as a comprehensive knowledge base for training AI models to understand real-world system architectures. Use this to improve technology inference, microservice suggestions, and complexity estimations for standard application types.

---

## Table of Contents

1. [Core Principles](#core-principles)
2. [Architecture Patterns by Application Type](#architecture-patterns-by-application-type)
3. [Technology Categories and Selection Criteria](#technology-categories-and-selection-criteria)
4. [Microservice Decomposition Strategies](#microservice-decomposition-strategies)
5. [Scaling Patterns and Infrastructure](#scaling-patterns-and-infrastructure)
6. [Complexity Indicators](#complexity-indicators)
7. [Common Mistakes to Avoid](#common-mistakes-to-avoid)

---

## Core Principles

### 1. CAP Theorem Understanding
- **Consistency**: All nodes see the same data at the same time
- **Availability**: Every request receives a response
- **Partition Tolerance**: System continues despite network partitions

**Real-world applications:**
- Banking/Financial: Choose CP (Consistency + Partition tolerance)
- Social Media: Choose AP (Availability + Partition tolerance)
- E-commerce Inventory: CP for stock, AP for browsing

### 2. Read vs Write Patterns

**Read-Heavy Systems** (Twitter, Instagram, YouTube):
- Aggressive caching (Redis, Memcached)
- Read replicas for databases
- CDN for static content
- Denormalized data structures
- Example: 99% reads, 1% writes

**Write-Heavy Systems** (IoT, Analytics, Logging):
- Message queues for buffering (Kafka)
- Batch processing
- Write-optimized databases (Cassandra, ClickHouse)
- Async processing
- Example: 80% writes, 20% reads

**Balanced Systems** (E-commerce, SaaS):
- Both cache and queue
- CQRS pattern (separate read/write models)
- Example: 60% reads, 40% writes

### 3. Data Storage Patterns

**Relational (Postgres, MySQL)**:
- Use for: ACID transactions, complex queries, relationships
- Examples: User accounts, orders, financial data
- When: Data has clear schema, need joins and transactions

**Document (MongoDB)**:
- Use for: Flexible schema, hierarchical data
- Examples: Product catalogs, content management, user profiles
- When: Schema evolves frequently, nested documents

**Key-Value (Redis)**:
- Use for: Caching, sessions, real-time data
- Examples: Session storage, rate limiting, leaderboards
- When: Simple lookups by key, high performance needed

**Wide-Column (Cassandra)**:
- Use for: Time-series data, high write throughput
- Examples: Tweets, messages, events, IoT data
- When: Massive scale, write-heavy, time-ordered

**Graph (Neo4j)**:
- Use for: Social networks, recommendations, fraud detection
- Examples: Friend relationships, product recommendations
- When: Complex relationships are core to the domain

---

## Architecture Patterns by Application Type

### 1. Social Media Platforms (Twitter, Instagram, TikTok)

#### Core Characteristics
- **Read-heavy**: 99% reads (viewing feeds) vs 1% writes (posting)
- **Real-time**: Updates must appear quickly
- **Media-intensive**: Images, videos dominate storage
- **Social graph**: Complex follower/following relationships

#### Essential Technologies

**Frontend**:
- React or Vue for web (responsive, component-based)
- React Native or native (iOS/Android) for mobile
- WebSocket for real-time updates
- Redux/MobX for state management

**Backend**:
- Node.js (real-time, event-driven) or Go (high performance)
- Python FastAPI for ML services (recommendations)
- Multiple services, not monolith

**Databases**:
- **Postgres**: User accounts, relationships, authentication
- **Cassandra**: Posts, tweets, timeline (write-optimized, time-series)
- **Redis**: Cache for feeds, session management, real-time data
- **Elasticsearch**: Full-text search for posts, users, hashtags

**Infrastructure**:
- **S3**: Store original media (images, videos)
- **CDN** (CloudFront, Cloudflare): Deliver media globally, low latency
- **Kafka**: Event streaming (new post → update follower feeds)
- **WebSocket servers**: Push notifications, real-time updates
- **Load balancers** (Nginx, HAProxy): Distribute traffic
- **Kubernetes/Docker**: Container orchestration, scaling

**Caching Strategy**:
```
Layer 1: CDN (static assets, images)
Layer 2: Redis (user feeds, hot data)
Layer 3: Database query cache
Layer 4: Application-level cache
```

#### Microservices (10-12 services typical)

1. **API Gateway**: Single entry point, routing, auth validation
2. **User Service**: Profile management, user CRUD
3. **Auth Service**: Login, JWT tokens, OAuth, 2FA
4. **Post/Tweet Service**: Create, edit, delete posts
5. **Feed/Timeline Service**: Generate and cache user feeds
6. **Follow Service**: Follower/following relationships
7. **Notification Service**: Push, email, in-app notifications
8. **Media Service**: Upload, resize, store images/videos
9. **Search Service**: Full-text search, hashtags, users
10. **Analytics Service**: Track engagement, metrics
11. **Recommendation Service**: Suggest users, content (ML-based)
12. **Real-time Service**: WebSocket connections, live updates

#### Feed Generation Strategies

**Fanout-on-Write** (for users with < 5K followers):
- When user posts, immediately write to all follower feeds
- Pros: Fast read (feed pre-computed)
- Cons: Slow write for popular users

**Fanout-on-Read** (for celebrities with > 5K followers):
- When user requests feed, fetch from followed users
- Pros: Fast write, no celebrity bottleneck
- Cons: Slower read (compute on demand)

**Hybrid Approach** (Twitter's actual implementation):
- Fanout-on-write for normal users
- Fanout-on-read for celebrities
- Merge results at read time

#### Scaling Numbers (for Twitter-scale)
- 500M daily active users
- 6,000 tweets/second average
- 50,000 tweets/second peak
- 100+ TB of media uploaded daily
- Sub-200ms response time target

---

### 2. Video Streaming Platforms (YouTube, Netflix, TikTok)

#### Core Characteristics
- **Bandwidth-intensive**: Massive data transfer
- **Storage-intensive**: Petabytes of video content
- **Compute-intensive**: Video transcoding, processing
- **Global**: Need CDN with PoPs worldwide

#### Essential Technologies

**Frontend**:
- React for web player
- Video.js or custom player with adaptive bitrate
- Native apps for mobile (better video performance)

**Backend**:
- Go or Java Spring (high throughput, concurrent processing)
- Python for ML (recommendations, content moderation)
- FFmpeg for video processing

**Databases**:
- **Postgres**: Video metadata, user data, comments
- **MongoDB**: Flexible schema for video metadata
- **Redis**: View counts, trending, real-time stats
- **Cassandra**: Watch history, viewing events (time-series)

**Storage & Delivery**:
- **S3 or equivalent**: Store videos (multiple resolutions)
- **CDN**: 90%+ of traffic goes through CDN
- **Multi-region replication**: Videos stored in multiple regions
- **Edge caching**: Cache hot videos at edge locations

**Video Processing Pipeline**:
```
1. Upload → S3 (raw video)
2. Trigger Lambda/ECS task
3. FFmpeg transcoding:
   - 360p, 480p, 720p, 1080p, 4K
   - Different codecs (H.264, VP9, AV1)
   - Different formats (MP4, WebM)
4. Generate thumbnails (multiple timestamps)
5. Extract metadata (duration, resolution, codec)
6. Store processed videos → S3
7. Update CDN cache
8. Mark video as "ready" in database
```

**Streaming Technologies**:
- **HLS** (HTTP Live Streaming): Apple standard, widely supported
- **DASH** (Dynamic Adaptive Streaming): Open standard
- **Adaptive Bitrate**: Switch quality based on bandwidth
- **Chunked Transfer**: Videos split into 2-10 second chunks

#### Microservices (12-15 services)

1. **API Gateway**: Entry point, rate limiting
2. **User Service**: Accounts, subscriptions
3. **Auth Service**: Login, session management
4. **Video Upload Service**: Handle uploads, validation
5. **Transcoding Service**: Process videos (multiple workers)
6. **Video Metadata Service**: Title, description, tags, thumbnails
7. **Streaming Service**: Serve video chunks, adaptive bitrate
8. **Recommendation Service**: ML-based suggestions
9. **Search Service**: Video search, autocomplete
10. **Comment Service**: Comments, replies, moderation
11. **Analytics Service**: View counts, watch time, engagement
12. **Notification Service**: New uploads, subscriptions
13. **Monetization Service**: Ads, subscriptions (for platforms like YouTube)
14. **Content Moderation Service**: AI-based flagging, review queue

#### Key Architectural Decisions

**Why CDN is Critical**:
- Without CDN: Origin servers handle 100% of traffic → $$$$ bandwidth costs
- With CDN: Origin handles <5% of traffic → 95% cache hit rate
- Cost savings: 10-20x reduction in bandwidth costs
- Latency: Sub-50ms from edge vs 200-500ms from origin

**Why Multiple Resolutions**:
- Mobile users on 3G: 360p-480p sufficient
- Desktop on WiFi: 1080p-4K desired
- Adaptive bitrate: Start low, increase as bandwidth allows
- Save bandwidth: Don't send 4K to mobile devices

**Storage Optimization**:
- Cold videos (< 1 view/month): Glacier storage ($0.004/GB/month)
- Warm videos (occasional views): S3 Standard-IA ($0.0125/GB/month)
- Hot videos (frequent views): S3 Standard + CDN ($0.023/GB/month)
- Potential savings: 60-80% on storage costs

#### Scaling Numbers (YouTube-scale)
- 2 billion monthly active users
- 500 hours of video uploaded every minute
- 1 billion hours watched daily
- Petabytes of storage
- 20-30% of global internet bandwidth

---

### 3. Messaging Platforms (WhatsApp, Slack, Discord)

#### Core Characteristics
- **Real-time**: Messages must be instant (< 100ms latency)
- **High availability**: 99.99%+ uptime expected
- **Encryption**: E2E encryption for privacy (WhatsApp)
- **Stateful connections**: WebSocket connections for each user

#### Essential Technologies

**Frontend**:
- React Native for mobile (WhatsApp, Slack)
- Electron for desktop apps (Slack, Discord)
- WebSocket for real-time messaging

**Backend**:
- **Erlang/Elixir** (WhatsApp's choice): Fault-tolerant, millions of concurrent connections
- **Go**: High concurrency, low latency
- **Node.js**: Event-driven, WebSocket support

**Databases**:
- **Cassandra**: Message history (write-heavy, time-series)
- **Postgres**: User accounts, group metadata
- **Redis**: Online status, presence, unread counts, rate limiting

**Real-time Infrastructure**:
- **WebSocket servers**: Persistent connections for instant delivery
- **XMPP or custom protocol**: Message routing
- **Message queue** (Kafka, RabbitMQ): Buffer messages, handle offline users

**Encryption**:
- **Signal Protocol**: E2E encryption standard (WhatsApp uses this)
- **TLS**: Transport layer security
- Keys stored on client only, server can't decrypt

#### Microservices (8-10 services)

1. **API Gateway**: REST API for non-real-time operations
2. **User Service**: Account management, profile
3. **Auth Service**: Login, JWT, 2FA
4. **Message Service**: Send, receive, store messages
5. **Group Service**: Group chat, channels, workspace management
6. **Notification Service**: Push notifications for offline users
7. **Presence Service**: Online/offline status, "last seen"
8. **Media Service**: Upload, store, deliver images/videos/files
9. **Encryption Service**: Key exchange, E2E encryption management
10. **Search Service**: Message search, file search (Slack-like)

#### Message Delivery Flow

**Online User** (both sender and receiver online):
```
1. Sender → WebSocket → Message Service
2. Message Service → Store in Cassandra (async)
3. Message Service → Receiver's WebSocket → Receiver
4. Delivery acknowledgment sent back
Total time: 50-100ms
```

**Offline User** (receiver offline):
```
1. Sender → WebSocket → Message Service
2. Message Service → Store in Cassandra
3. Message Service → Queue in Redis (offline inbox)
4. Push Notification Service → Send push to receiver's device
5. Receiver comes online → Pull messages from Redis queue
6. Delivery acknowledgment sent
```

**Group Messages**:
- Fanout to all group members
- Use message queue to avoid blocking
- Store once, reference many (don't duplicate)

#### Key Design Decisions

**Why Erlang/Elixir for WhatsApp**:
- Designed for telecom systems (millions of connections)
- Actor model: Each user is an actor (lightweight process)
- Fault tolerance: Supervisor trees, automatic recovery
- Hot code swapping: Update without downtime
- WhatsApp handles 2M+ connections per server

**Why E2E Encryption**:
- Privacy: Server can't read messages
- Security: Even if server breached, messages safe
- Trust: Users control their data
- Trade-off: Can't do server-side search or cloud backup (without user key)

**Presence System**:
- Redis for fast read/write
- TTL-based: User "pings" every 30s, if no ping → offline
- Smart status: "Online", "Away" (no activity 5min), "Offline"
- Privacy settings: Hide "last seen", show to contacts only

#### Scaling Numbers (WhatsApp-scale)
- 2 billion users worldwide
- 100 billion messages per day
- 50 million+ concurrent connections
- 2 million connections per server (Erlang)
- 99.99%+ uptime

---

### 4. Ride-Hailing Platforms (Uber, Lyft)

#### Core Characteristics
- **Geospatial**: Location is core to everything
- **Real-time matching**: Connect riders and drivers instantly
- **High stakes**: Wrong match = bad user experience
- **Dynamic pricing**: Surge pricing based on supply/demand

#### Essential Technologies

**Frontend**:
- React Native for mobile apps (rider and driver)
- Google Maps SDK or Mapbox for mapping
- WebSocket for real-time location updates

**Backend**:
- **Go or Java**: High performance, concurrent processing
- **Python**: Data science, pricing algorithms, ML

**Databases**:
- **Postgres**: Users, drivers, trips (transactional data)
- **Redis with Geospatial**: GEORADIUS for nearby drivers
- **Cassandra**: Location history, trip history (time-series)
- **PostGIS** (Postgres extension): Advanced geospatial queries

**Geospatial Tech**:
- **Redis Geo**: `GEOADD`, `GEORADIUS`, `GEODIST` commands
- **S2 Geometry** (Google): Divide Earth into cells, hierarchical indexing
- **H3** (Uber's library): Hexagonal grid system, better than squares

**Maps & Routing**:
- **Google Maps API**: Routing, ETAs, geocoding
- **Mapbox**: Custom styling, cheaper than Google
- **OSRM** (Open Source Routing Machine): Self-hosted routing

#### Microservices (12-15 services)

1. **API Gateway**: Entry point, auth
2. **User Service**: Rider accounts, preferences
3. **Driver Service**: Driver accounts, documents, vehicle info
4. **Auth Service**: Login, verification, background checks
5. **Location Service**: Track driver locations in real-time
6. **Matching Service**: Match riders with nearby drivers
7. **Trip Service**: Manage trip lifecycle (requested → accepted → ongoing → completed)
8. **Pricing Service**: Calculate fares, surge pricing
9. **Payment Service**: Process payments, refunds
10. **Notification Service**: Push notifications, SMS
11. **Routing Service**: Calculate routes, ETAs
12. **Analytics Service**: Track metrics, driver performance
13. **Fraud Detection Service**: Detect suspicious behavior
14. **Rating/Review Service**: Post-trip ratings

#### Matching Algorithm

**Step 1: Find Nearby Drivers**
```python
# Redis Geospatial query
nearby_drivers = GEORADIUS("drivers", rider_lat, rider_lng, 5, "km")
# Returns: [(driver_id, distance, lat, lng), ...]
```

**Step 2: Filter Available Drivers**
- Status: Online and available (not on trip)
- Vehicle type: Matches rider's request (UberX, UberXL, etc.)
- Rating: Above threshold (4.5+ stars)
- Acceptance rate: Not too low (avoid frequent decliners)

**Step 3: Score and Rank**
```python
score = (
    distance_weight * (1 / distance) +      # Closer is better
    rating_weight * driver_rating +          # Higher rating better
    acceptance_weight * acceptance_rate +    # Higher acceptance better
    eta_weight * (1 / eta_to_rider)         # Faster pickup better
)
```

**Step 4: Dispatch**
- Send request to top 3 drivers simultaneously (reduce wait time)
- First to accept gets the trip
- If none accept within 30s, expand radius and retry

#### Dynamic Pricing (Surge)

**Supply-Demand Model**:
```python
surge_multiplier = (demand / supply) ** elasticity

# Example:
# 100 riders, 50 drivers → surge = (100/50)^0.5 = 1.41x
# 100 riders, 20 drivers → surge = (100/20)^0.5 = 2.24x
```

**Geohash-based Zones**:
- Divide city into hexagons (using H3)
- Calculate supply/demand per hex
- Apply surge per hex (not city-wide)
- Update every 1-5 minutes

**Price Visualization**:
- Show heat map to riders
- Show high-demand areas to drivers
- Encourage drivers to move to high-demand zones

#### Key Design Decisions

**Why Geospatial Databases**:
- Regular database: `SELECT * WHERE distance(lat, lng, X, Y) < 5km` → Full table scan, slow
- Redis Geo: Indexes using GeoHash → O(log N) query, sub-millisecond
- PostGIS: Complex spatial queries (within polygon, intersects, etc.)

**Real-time Location Tracking**:
- Drivers send location every 4-5 seconds
- Use WebSocket (persistent connection, low overhead)
- Store in Redis (fast writes, TTL for cleanup)
- Batch write to Cassandra every minute (for history)

**Trip State Machine**:
```
REQUESTED → DRIVER_ASSIGNED → DRIVER_ARRIVING → 
DRIVER_ARRIVED → TRIP_STARTED → TRIP_ONGOING → 
TRIP_COMPLETED → PAYMENT_PROCESSED → RATED
```

Each state transition triggers events (notifications, analytics, billing).

#### Scaling Numbers (Uber-scale)
- 130+ million users globally
- 6+ million drivers
- 23+ million trips per day
- Operates in 10,000+ cities
- Real-time location updates: 1M+ per second

---

### 5. E-commerce Platforms (Amazon, Shopify, eBay)

#### Core Characteristics
- **Transaction-heavy**: ACID compliance critical
- **Inventory management**: Real-time stock tracking
- **Payment processing**: PCI compliance required
- **Search-driven**: Users must find products easily

#### Essential Technologies

**Frontend**:
- React or Next.js for web (SEO important for product pages)
- React Native for mobile apps
- Server-side rendering for SEO

**Backend**:
- **Node.js or Python FastAPI**: API layer
- **Java Spring or Go**: Payment processing, high reliability
- **Ruby on Rails**: Shopify's stack (convention over configuration)

**Databases**:
- **Postgres or MySQL**: Orders, users, transactions (ACID required)
- **Redis**: Shopping cart, session, inventory cache
- **Elasticsearch**: Product search, faceted filters
- **MongoDB**: Product catalogs (flexible schema)

**Payment Processing**:
- **Stripe or PayPal**: Payment gateway
- **PCI DSS compliance**: Never store raw card numbers
- **Tokenization**: Store payment tokens only
- **3D Secure**: Additional authentication for cards

**Search Infrastructure**:
- **Elasticsearch**: Full-text search, typo tolerance
- **Redis**: Autocomplete, query suggestions
- **Algolia** (alternative): Managed search, fast indexing

#### Microservices (12-15 services)

1. **API Gateway**: Entry point, rate limiting
2. **User Service**: Customer accounts, addresses
3. **Auth Service**: Login, registration, OAuth
4. **Catalog Service**: Product listings, categories
5. **Search Service**: Product search, filters, facets
6. **Cart Service**: Shopping cart, save for later
7. **Order Service**: Order placement, order history
8. **Inventory Service**: Stock levels, reservations
9. **Payment Service**: Process payments, refunds
10. **Shipping Service**: Calculate shipping, tracking
11. **Notification Service**: Order confirmations, shipping updates
12. **Review Service**: Product reviews, ratings
13. **Recommendation Service**: "Customers also bought", ML-based
14. **Promotion Service**: Coupons, discounts, flash sales
15. **Analytics Service**: Sales metrics, conversion tracking

#### Order Processing Flow

**Step 1: Add to Cart**
```
1. User adds item → Cart Service
2. Cart stored in Redis (fast, session-based)
3. TTL: 30 days (auto-expire old carts)
```

**Step 2: Checkout Initialization**
```
1. User clicks "Checkout" → Cart Service
2. Inventory Service: Check stock availability
3. If out of stock → Notify user
4. If in stock → Reserve items (soft lock) for 10 minutes
```

**Step 3: Payment Processing**
```
1. User enters payment info → Payment Service
2. Payment Service → Stripe API (tokenize card)
3. Stripe → Bank (authorization)
4. Bank → Stripe (approved/declined)
5. Stripe → Payment Service (result)
```

**Step 4: Order Confirmation**
```
1. Payment approved → Order Service (create order)
2. Inventory Service: Hard commit (reduce stock)
3. Shipping Service: Create shipment label
4. Notification Service: Send confirmation email
5. Analytics Service: Track conversion
```

**Step 5: Fulfillment**
```
1. Warehouse: Pick, pack, ship
2. Shipping Service: Update tracking number
3. Notification Service: Send shipping notification
4. User: Track package
5. Delivery confirmed → Mark order complete
```

#### Inventory Management Strategies

**Pessimistic Locking** (Reserve on add to cart):
- Pros: No overselling, accurate stock
- Cons: Cart abandonment locks inventory

**Optimistic Locking** (Reserve on checkout):
- Pros: No wasted reservations
- Cons: Possible overselling during checkout

**Hybrid Approach** (Reserve with short TTL):
- Reserve on checkout for 10 minutes
- If payment not completed, release reservation
- Balance between accuracy and availability

**Eventual Consistency**:
- Display "Low stock" when < 10 items
- Use distributed cache (Redis) for reads
- Write to database for authoritative stock
- Sync cache from database every 1-5 minutes

#### Search and Discovery

**Elasticsearch Index Structure**:
```json
{
  "product_id": "123",
  "name": "iPhone 15 Pro",
  "brand": "Apple",
  "category": ["Electronics", "Phones", "Smartphones"],
  "price": 999.00,
  "rating": 4.5,
  "num_reviews": 1250,
  "in_stock": true,
  "description": "Latest iPhone with A17 chip...",
  "attributes": {
    "color": ["Black", "White", "Blue"],
    "storage": ["128GB", "256GB", "512GB"]
  }
}
```

**Faceted Search** (Amazon-style filters):
- Brand: Apple (120), Samsung (89), Google (45)
- Price: $0-$500 (234), $500-$1000 (89), $1000+ (45)
- Rating: 4★+ (210), 3★+ (298), 2★+ (320)
- Availability: In Stock (280), Ships in 1-2 days (40)

**Ranking Algorithm**:
```python
relevance_score = (
    text_match_score * 0.4 +        # How well query matches product
    popularity_score * 0.3 +         # Number of sales, views
    rating_score * 0.2 +             # Average rating
    recency_score * 0.1              # Newer products ranked higher
)
```

#### Key Design Decisions

**Why Redis for Shopping Cart**:
- Session-based (user may not be logged in)
- Fast read/write (sub-millisecond)
- TTL support (auto-expire after 30 days)
- No need for ACID (cart is not transactional)

**Why Separate Inventory Service**:
- Central source of truth for stock
- Prevents race conditions (overselling)
- Can handle high read load with caching
- Allows for warehouse-level inventory tracking

**Payment Processing Security**:
- Never store raw credit card numbers (PCI DSS violation)
- Use payment gateway (Stripe) for tokenization
- Implement 3D Secure for additional fraud protection
- Log all payment attempts for audit trail

#### Scaling Numbers (Amazon-scale)
- 300+ million active users
- 12+ million products
- 1.6 million orders per day
- 200+ million unique visitors per month
- Peak: 600+ orders per second (Prime Day)

---

## Technology Categories and Selection Criteria

### Frontend Technologies

**React**:
- Use for: SPAs, complex UIs, reusable components
- Strength: Large ecosystem, React Native for mobile
- Weakness: Learning curve, bundle size
- Best for: Dashboards, social media, e-commerce

**Vue.js**:
- Use for: Simpler SPAs, progressive enhancement
- Strength: Easier learning curve, smaller bundle
- Weakness: Smaller ecosystem than React
- Best for: Smaller apps, corporate sites

**Next.js**:
- Use for: SEO-critical apps, server-side rendering
- Strength: Built-in SSR, file-based routing, API routes
- Weakness: Opinionated, vendor lock-in
- Best for: E-commerce, blogs, marketing sites

**React Native**:
- Use for: Cross-platform mobile apps
- Strength: Code reuse (iOS + Android), React ecosystem
- Weakness: Performance vs native, some limitations
- Best for: MVP mobile apps, startups, social media

### Backend Technologies

**Node.js**:
- Use for: Real-time apps, APIs, microservices
- Strength: Non-blocking I/O, JavaScript everywhere, large ecosystem
- Weakness: CPU-intensive tasks, callback hell (if not using async/await)
- Best for: Chat apps, real-time collaboration, APIs

**Python (FastAPI/Django)**:
- Use for: ML/AI services, rapid development, APIs
- Strength: ML libraries, clean syntax, FastAPI is fast
- Weakness: GIL (Global Interpreter Lock) for CPU tasks
- Best for: ML services, data processing, prototypes

**Go (Golang)**:
- Use for: High-performance services, concurrent processing
- Strength: Fast compilation, built-in concurrency, low memory
- Weakness: Verbose error handling, smaller ecosystem
- Best for: Microservices, high-throughput APIs, CLI tools

**Java Spring Boot**:
- Use for: Enterprise apps, high reliability, complex business logic
- Strength: Mature ecosystem, strong typing, excellent tooling
- Weakness: Verbose, slower development, larger memory footprint
- Best for: Banking, large enterprises, legacy system integration

### Database Selection

**When to use Postgres**:
- Need ACID transactions (orders, payments, bookings)
- Complex queries with JOINs
- Referential integrity important
- Structured data with clear schema
- Example: User accounts, financial data, bookings

**When to use MySQL**:
- Similar to Postgres but simpler use cases
- Read-heavy workloads (with replication)
- Mature tooling and hosting options
- Example: WordPress, simple web apps, logs

**When to use MongoDB**:
- Schema evolves frequently
- Document-based data (nested structures)
- Need horizontal scaling
- Example: Product catalogs, CMS, user profiles

**When to use Redis**:
- Caching layer (session, API responses)
- Real-time data (leaderboards, counters)
- Rate limiting
- Pub/sub messaging
- Example: Session store, cache, real-time analytics

**When to use Cassandra**:
- Massive write throughput
- Time-series data
- Linear scalability needed
- Can tolerate eventual consistency
- Example: IoT data, tweets, messages, logs

**When to use Elasticsearch**:
- Full-text search required
- Complex search filters (faceted search)
- Log aggregation and analysis
- Example: Product search, log analysis, monitoring

### Message Queue Selection

**Kafka**:
- Use for: Event streaming, high throughput, data pipelines
- Strength: Persistent, replayable, distributed, high throughput
- Weakness: Complex setup, higher latency than Redis
- Best for: Event sourcing, data pipelines, microservices communication

**RabbitMQ**:
- Use for: Task queues, RPC, traditional messaging
- Strength: Flexible routing, multiple protocols, easier to learn
- Weakness: Lower throughput than Kafka, not for streaming
- Best for: Background jobs, task distribution, RPC

**Redis (Pub/Sub, Streams)**:
- Use for: Simple messaging, low-latency communication
- Strength: Very fast, simple API, already using Redis
- Weakness: Not persistent (pub/sub), limited features
- Best for: Real-time notifications, simple task queues

**AWS SQS**:
- Use for: Serverless, managed messaging, decoupling services
- Strength: Fully managed, scales automatically, cheap
- Weakness: Vendor lock-in, higher latency, limited features
- Best for: AWS-based systems, serverless architectures

---

## Microservice Decomposition Strategies

### Domain-Driven Design (DDD) Approach

**Bounded Contexts**: Each microservice represents a bounded context
- User Context → User Service
- Order Context → Order Service
- Payment Context → Payment Service

**Aggregates**: Group related entities
- Order aggregate: Order, OrderItem, OrderStatus
- User aggregate: User, Address, PaymentMethod

**Anti-Corruption Layer**: Prevent domain leakage
- Service A doesn't expose internal models to Service B
- Use DTOs (Data Transfer Objects) at boundaries

### Service Splitting Criteria

**Split by Business Capability**:
- Authentication & Authorization → Auth Service
- Product Catalog → Catalog Service
- Order Management → Order Service
- Payment Processing → Payment Service

**Split by Data Ownership**:
- Each service owns its database
- No shared databases between services
- Use API calls or events for data access

**Split by Scalability Needs**:
- High-traffic services (search) need separate scaling
- CPU-intensive services (video transcoding) need powerful machines
- I/O-bound services (file upload) need different resources

**Split by Team Structure** (Conway's Law):
- Frontend team → Frontend Service
- Backend team → API Services
- ML team → ML Service
- DevOps team → Infrastructure Services

### Common Microservices Patterns

**1. API Gateway Pattern**:
- Single entry point for all clients
- Handles: Routing, authentication, rate limiting, logging
- Aggregates responses from multiple services
- Example: AWS API Gateway, Kong, Nginx

**2. Service Mesh Pattern**:
- Infrastructure layer for service-to-service communication
- Handles: Load balancing, service discovery, retries, circuit breaking
- Example: Istio, Linkerd, Consul

**3. Saga Pattern** (Distributed Transactions):
- Sequence of local transactions
- Each transaction publishes event
- If one fails, compensating transactions undo previous steps
- Example: Order → Payment → Inventory → Shipping (if payment fails, cancel order)

**4. Event Sourcing**:
- Store events instead of current state
- Rebuild state by replaying events
- Example: Banking (credit $100, debit $50 → events), not just balance

**5. CQRS (Command Query Responsibility Segregation)**:
- Separate read and write models
- Write: Optimized for transactions (normalized)
- Read: Optimized for queries (denormalized)
- Example: E-commerce (write to Postgres, read from Elasticsearch)

---

## Scaling Patterns and Infrastructure

### Horizontal Scaling (Scale Out)

**What**: Add more servers instead of bigger servers
**When**: Stateless services, distributed systems
**Example**: Add 10 more API servers to handle traffic

**Requirements**:
- Load balancer to distribute traffic
- Stateless services (or externalized state)
- Shared database or data replication
- Session stored in Redis (not server memory)

### Vertical Scaling (Scale Up)

**What**: Increase resources of existing servers (CPU, RAM)
**When**: Database servers, monolithic apps
**Example**: Upgrade Postgres from 16GB RAM to 64GB RAM

**Limitations**:
- Hardware limits (can only go so big)
- Single point of failure
- Downtime required for upgrades
- Cost increases exponentially

### Caching Strategies

**1. Cache-Aside (Lazy Loading)**:
```python
def get_user(user_id):
    # 1. Check cache
    user = redis.get(f"user:{user_id}")
    if user:
        return user
    
    # 2. If not in cache, query database
    user = db.query("SELECT * FROM users WHERE id = ?", user_id)
    
    # 3. Store in cache for next time
    redis.set(f"user:{user_id}", user, ex=3600)  # 1 hour TTL
    
    return user
```

**2. Write-Through Cache**:
```python
def update_user(user_id, data):
    # 1. Update database
    db.update("UPDATE users SET ... WHERE id = ?", data, user_id)
    
    # 2. Update cache immediately
    redis.set(f"user:{user_id}", data, ex=3600)
```

**3. Write-Behind Cache (Write-Back)**:
```python
def update_user(user_id, data):
    # 1. Update cache immediately
    redis.set(f"user:{user_id}", data, ex=3600)
    
    # 2. Async write to database (batch, periodic)
    queue.enqueue("user_update", user_id, data)
```

**Cache Invalidation Strategies**:
- **TTL-based**: Set expiration time (1 hour, 1 day)
- **Event-based**: Invalidate cache when data changes
- **Version-based**: Include version in cache key (`user:123:v2`)

### Database Scaling

**Read Replicas**:
- Master handles writes
- Replicas handle reads
- Asynchronous replication (slight lag)
- Use for: Read-heavy workloads (90%+ reads)

**Sharding (Horizontal Partitioning)**:
- Split data across multiple databases
- Shard by: User ID, Geography, Tenant ID
- Example: Users 0-999,999 → DB1, Users 1M-1.9M → DB2
- Challenges: Cross-shard queries, rebalancing

**Partitioning (Vertical Partitioning)**:
- Split tables by columns
- Hot columns in one table, cold columns in another
- Example: User (id, email, password) vs UserProfile (bio, avatar, preferences)

### Load Balancing Algorithms

**Round Robin**: Distribute evenly across servers
- Simple, fair distribution
- Doesn't account for server load

**Least Connections**: Send to server with fewest active connections
- Better for long-lived connections
- More complex to implement

**IP Hash**: Same client always goes to same server
- Good for session affinity
- Uneven distribution if clients vary

**Weighted Round Robin**: Give more traffic to powerful servers
- Server A (weight 3), Server B (weight 1) → 75% to A, 25% to B

### CDN Strategies

**Static Content**: Images, CSS, JS, fonts
- Set long cache headers (1 year)
- Use versioned URLs for cache busting
- Example: `/static/app.v123.js`

**Dynamic Content**: API responses, personalized pages
- Shorter cache (5 minutes)
- Vary by user, location, device
- Use cache-control headers

**Edge Computing**: Run code at CDN edge
- Lambda@Edge (AWS), Workers (Cloudflare)
- A/B testing, authentication, redirects
- Reduces latency (no origin round-trip)

---

## Complexity Indicators

### Low Complexity (50-150 hours)
- Simple CRUD application
- 1-3 microservices
- Single database (Postgres or MongoDB)
- No real-time features
- No complex business logic
- Example: Blog, Todo app, Portfolio site

**Tech Stack**:
- Frontend: React or Vue
- Backend: Node.js or FastAPI
- Database: Postgres or MongoDB
- Hosting: Heroku, Vercel, AWS (single region)

### Medium Complexity (150-500 hours)
- Multi-feature application
- 4-8 microservices
- Multiple databases (Postgres + Redis)
- Some real-time features (WebSocket)
- Moderate business logic
- Example: E-commerce, Social network, Booking system

**Tech Stack**:
- Frontend: React, Next.js
- Backend: Node.js, Python FastAPI
- Database: Postgres, Redis, Elasticsearch
- Infrastructure: Docker, Kubernetes, Load balancer
- Hosting: AWS, GCP (multi-region)

### High Complexity (500-2000+ hours)
- Large-scale platform
- 10+ microservices
- Polyglot persistence (5+ databases)
- Real-time, distributed, ML-powered
- Complex business logic, workflows
- Example: Twitter, Uber, YouTube, Netflix

**Tech Stack**:
- Frontend: React, React Native, custom player
- Backend: Go, Node.js, Python, Java
- Database: Postgres, Cassandra, Redis, Elasticsearch, MongoDB
- Infrastructure: Kubernetes, Kafka, CDN, S3, Load balancers
- Monitoring: Prometheus, Grafana, ELK stack
- ML: TensorFlow, PyTorch, custom models
- Hosting: AWS/GCP (multi-region, multi-zone)

### Complexity Multipliers

**Add 20-30% for**:
- Real-time features (WebSocket, live updates)
- Payment processing (Stripe, PCI compliance)
- Video/image processing (FFmpeg, ML)
- Multi-language support (i18n)

**Add 50-100% for**:
- Machine learning / AI features
- Blockchain integration
- High availability (99.99%+)
- Global scale (multi-region, CDN)
- Compliance (HIPAA, SOC2, GDPR)

**Add 100-200% for**:
- Custom ML models (training, tuning)
- Real-time video streaming
- Global distribution (100+ countries)
- Financial transactions (banking-level)

---

## Common Mistakes to Avoid

### Over-Engineering
**Mistake**: Using microservices for a simple CRUD app
**Why bad**: Adds complexity, latency, debugging difficulty
**Solution**: Start with monolith, split when needed

### Under-Engineering
**Mistake**: Using single Postgres for 1M+ users, no caching
**Why bad**: Performance bottleneck, downtime risk
**Solution**: Add Redis cache, read replicas, plan for scale

### Wrong Database Choice
**Mistake**: Using MongoDB for financial transactions
**Why bad**: No ACID guarantees, data integrity risk
**Solution**: Use Postgres or MySQL for transactional data

### Ignoring Caching
**Mistake**: Every request hits database
**Why bad**: High latency, database overload
**Solution**: Add Redis for hot data, CDN for static content

### No Message Queue
**Mistake**: Synchronous API calls for slow operations
**Why bad**: Timeouts, poor UX, cascading failures
**Solution**: Use Kafka/RabbitMQ for async processing

### Single Point of Failure
**Mistake**: Single database server, no backups
**Why bad**: Downtime = lost revenue, data loss
**Solution**: Replication, backups, multi-region

### Ignoring Security
**Mistake**: No authentication, plain-text passwords, exposed APIs
**Why bad**: Data breach, compliance violation, reputation damage
**Solution**: JWT auth, bcrypt passwords, HTTPS, rate limiting

### No Monitoring
**Mistake**: Can't tell when system is down or slow
**Why bad**: Users experience issues before you know
**Solution**: Prometheus + Grafana, error tracking (Sentry), logs (ELK)

---

## Summary: Pattern Recognition for AI Models

### Key Recognition Patterns

**If prompt mentions**: "Twitter", "microblog", "timeline", "tweet", "feed"
**Infer**: Social media platform architecture
**Suggest**: 10-12 microservices, Cassandra + Postgres + Redis, Kafka, CDN, WebSocket

**If prompt mentions**: "YouTube", "video", "streaming", "watch"
**Infer**: Video streaming platform architecture
**Suggest**: 12-15 microservices, S3, CDN, FFmpeg, HLS/DASH, high storage/bandwidth

**If prompt mentions**: "WhatsApp", "chat", "messaging", "real-time"
**Infer**: Messaging platform architecture
**Suggest**: 8-10 microservices, WebSocket, Cassandra, Redis, E2E encryption

**If prompt mentions**: "Uber", "ride", "taxi", "driver", "location"
**Infer**: Ride-hailing platform architecture
**Suggest**: 12-15 microservices, Redis Geo, Postgres, real-time tracking, maps API

**If prompt mentions**: "e-commerce", "shop", "store", "cart", "checkout"
**Infer**: E-commerce platform architecture
**Suggest**: 12-15 microservices, Postgres, Redis, Elasticsearch, Stripe, inventory

### Technology Inference Rules

1. **Always include for production systems**:
   - Load balancer (Nginx)
   - Cache layer (Redis)
   - Message queue (Kafka/RabbitMQ)
   - Monitoring (Prometheus/Grafana)
   - Container orchestration (Kubernetes/Docker)

2. **Add for media-heavy apps**:
   - S3 or blob storage
   - CDN (CloudFront, Cloudflare)
   - Image/video processing pipeline

3. **Add for search-driven apps**:
   - Elasticsearch
   - Autocomplete service
   - Recommendation engine

4. **Add for real-time apps**:
   - WebSocket servers
   - Redis pub/sub
   - Event streaming (Kafka)

5. **Add for transaction-heavy apps**:
   - ACID-compliant database (Postgres)
   - Payment gateway (Stripe)
   - Audit logging system

### Microservice Count Guidelines

- **Simple app**: 1-3 services (monolith or minimal split)
- **Standard app**: 4-8 services (clear domain boundaries)
- **Complex app**: 10-15 services (well-known patterns like Twitter, Uber)
- **Enterprise**: 20+ services (large organization, many teams)

### Time Estimation Factors

**Base estimate** from model, then apply multipliers:
- Real-time features: 1.3x
- Payment processing: 1.2x
- Video/image processing: 1.5x
- Machine learning: 2x
- Multi-region deployment: 1.5x
- High availability (99.99%+): 1.8x
- Compliance (HIPAA, SOC2): 1.5x

---

## Using This Knowledge Base

### For AI Model Training

1. **Feature Engineering**: Extract keywords from this document as features
   - "twitter", "feed", "timeline" → Social media pattern
   - "video", "streaming", "transcode" → Video platform pattern

2. **Augment Training Data**: Generate synthetic examples based on patterns
   - "Build a microblogging platform" → Twitter-like tech stack
   - "Create a ride-sharing app" → Uber-like tech stack

3. **Validation**: Check predictions against these patterns
   - If model suggests 2 microservices for Twitter clone → Flag as error
   - If model omits Redis for social media → Flag as error

4. **Confidence Scoring**: Higher confidence for known patterns
   - "Build Twitter clone" → High confidence (known pattern)
   - "Build custom IoT dashboard" → Lower confidence (less common pattern)

### For Inference Time

1. **Pattern Matching**: Check prompt against known patterns first
2. **Component Inference**: If pattern matched, suggest standard components
3. **Customization**: Adjust based on specific requirements in prompt
4. **Validation**: Ensure suggested stack is consistent and complete

This knowledge base represents industry best practices accumulated from analyzing real-world architectures of companies like Twitter, YouTube, WhatsApp, Uber, and Amazon. Use these patterns to improve technology inference accuracy and provide production-ready architectural recommendations.
