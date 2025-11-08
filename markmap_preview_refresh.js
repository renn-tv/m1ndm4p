const source = new EventSource('http://localhost:8765/preview-events');
source.onmessage = () => location.reload();
