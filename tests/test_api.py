from fastapi.testclient import TestClient
from app import app
client=TestClient(app)
def test_health(): assert client.get('/health').status_code==200
def test_spam_scores_higher():
 a=client.post('/predict',json={'text':'URGENT! You have won a free prize. Call now to claim cash'}).json()['spam_probability']
 b=client.post('/predict',json={'text':'Hi, are we still meeting for lunch today?'}).json()['spam_probability']
 assert a>b
def test_empty_and_missing_text_rejected():
 assert client.post('/predict',json={'text':''}).status_code==422
 assert client.post('/predict',json={}).status_code==422
def test_metrics_increment():
 before=client.get('/metrics').json()['request_count']
 client.post('/predict',json={'text':'hello'})
 assert client.get('/metrics').json()['request_count']==before+1
