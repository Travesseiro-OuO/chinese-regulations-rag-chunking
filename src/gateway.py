"""Fixed-destination DeepSeek relay. No arbitrary proxying; secret never logged."""
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from pathlib import Path
import requests,json
KEY=Path('/run/secrets/deepseek_key').read_text().strip()
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*args):pass
 def do_GET(self):
  if self.path!='/health':self.send_error(404);return
  self.send_response(200);self.end_headers();self.wfile.write(b'ok')
 def do_POST(self):
  if self.path!='/chat/completions':self.send_error(404);return
  n=int(self.headers.get('Content-Length',0))
  if not 0<n<100000:self.send_error(413);return
  try:
   payload=json.loads(self.rfile.read(n))
   if payload.get('model') not in ['deepseek-chat','deepseek-reasoner']:self.send_error(400);return
   r=requests.post('https://api.deepseek.com/chat/completions',json=payload,headers={'Authorization':'Bearer '+KEY},timeout=120)
   self.send_response(r.status_code);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(r.content)
  except Exception as e:
   self.send_error(502,type(e).__name__)
ThreadingHTTPServer(('0.0.0.0',8080),Handler).serve_forever()
