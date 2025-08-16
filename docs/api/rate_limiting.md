# Rate Limiting

The rate limiting system protects the Greyhound Racing Predictor's API from abuse and ensures fair use among all clients.

## Rate Limiting Strategy

### Global Limits
- **Global Request Cap**: Maximum number of requests per minute for whole service
- **Per-IP Limits**: Restriction on a per-IP basis to prevent abuse

## Hierarchical Limits

### Endpoint-Specific Limits
- **High-Traffic Endpoints**
  - `/api/predict_single_race`: Limited to 100 requests per minute
  - `/api/predict_stream`: Limited to 50 concurrent connections
  - `/api/dogs/search`: Limited to 200 requests per minute

### User-Based Limits
- **Role-Based Access Control**: Different groups have specific limits
- **Admin**: Higher limits for administrative operations
- **User**: Standard limits for regular usage
- **Viewer**: Read-only limits with higher read access

## Implementation Details

### Middleware Integration
- Utilizes Flask's middleware capabilities to intercept requests

```python
def rate_limit(view_function):
    @functools.wraps(view_function)
    def wrapped_function(*args, **kwargs):
        limit = request_limiter.get_limit(request.remote_addr)
        if not limit.allow_request():
            return jsonify({'error': 'Too many requests'}), 429
        return view_function(*args, **kwargs)
    return wrapped_function
```

### Configuration
- **Custom Configurations**: Editable configs stored in `rate_limiting.yaml`

## Exceeding Limits

### Clients Exceeding Limits
- **429 Response Code**: Standard response for exceeding limits
- **Retry-After Header**: Advises when to retry request

### Handling Violations
- **Logging Violations**: Document every rate limit infringement
- **Contact User**: Notify offenders, escalate repeat violators

## Reporting and Monitoring

### Dashboard
- **Real-Time Metrics**: Display current API usage statistics
- **Historical Data**: Keep track of long-term usage patterns

### Alerts
- **Threshold Alerts**: Notify admins when limits are frequently exceeded
- **Usage Spikes**: Real-time alerts for sudden usage increase

## Performance Optimizations
- **Adaptive Rate Limits**: Dynamically adjusting based on server load
- **Caching Optimizations**: Use Redis-based caching for fast limit checks

By implementing robust rate limiting, the system ensures equitable access and prevents any single user from overwhelming the service.
