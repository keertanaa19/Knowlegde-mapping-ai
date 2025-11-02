# app.py
from flask import Flask, request, jsonify, render_template_string, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3, jwt, io, csv, time
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from datetime import datetime as dt

# -------------------- CONFIG --------------------
DB = "users.db"
SECRET_KEY = "mysecretkey"
JWT_ALGO = "HS256"

app = Flask(__name__)
app.secret_key = SECRET_KEY

# -------------------- MODELS --------------------
nlp = spacy.load("en_core_web_sm")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

semantic_search_data = []
FEEDBACK_LIST = []
PROCESSING_DAILY = {i: 0 for i in range(7)}
STATS = {"files_uploaded": 0, "triples_total": 0, "unique_nodes": 0}


# -------------------- DB INIT --------------------
def init_db():
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        email TEXT,
        password_hash TEXT
    )""")
    conn.commit()
    conn.close()
init_db()


# -------------------- HELPERS --------------------
def recalc_node_counts():
    nodes = set()
    for t in semantic_search_data:
        nodes.add(t["subject"])
        nodes.add(t["object"])
    STATS["unique_nodes"] = len(nodes)
    STATS["triples_total"] = len(semantic_search_data)

def bump_processing(n=1):
    wd = dt.utcnow().weekday()
    PROCESSING_DAILY[wd] = PROCESSING_DAILY.get(wd, 0) + n

def compute_avg_rating():
    if not FEEDBACK_LIST:
        return 0
    return round(sum(f["rating"] for f in FEEDBACK_LIST) / len(FEEDBACK_LIST), 2)


# -------------------- HTML BASE --------------------
BASE_HEAD = """
<!doctype html>
<html><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>KnowMap with AI</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
<link href="https://unpkg.com/vis-network@9.1.2/dist/vis-network.min.css" rel="stylesheet">
<style>
body{background:#f6f7fb;font-family:Inter,system-ui,Segoe UI,Roboto,Helvetica,Arial,sans-serif;}
.navbar{box-shadow:0 3px 10px rgba(0,0,0,0.05);}
.card{border-radius:10px;}
#graph{height:420px;background:#fff;border-radius:8px;padding:8px;border:1px solid #e6e9ef;}
.star{cursor:pointer;font-size:22px;color:#ccc;}
.star.selected{color:#ffbf00;}
.dashboard-scroll{max-height:80vh;overflow-y:auto;}
.hero{padding:100px 0;text-align:center;}
</style>
</head><body>
"""

NAVBAR = """
<nav class="navbar navbar-expand-lg bg-white">
  <div class="container">
    <a class="navbar-brand fw-bold" href="/">KnowMap with AI</a>
    <div class="collapse navbar-collapse">
      <ul class="navbar-nav me-auto">
        <li class="nav-item"><a class="nav-link" href="/upload">Upload File</a></li>
        <li class="nav-item"><a class="nav-link" href="/semantic">Semantic Search</a></li>
        <li class="nav-item"><a class="nav-link" href="/dashboard">Dashboard</a></li>
      </ul>
      <div id="navUser" class="d-flex">
        <a class="btn btn-outline-secondary btn-sm me-2" href="/login">Login</a>
        <a class="btn btn-outline-danger btn-sm" href="/logout">Logout</a>
      </div>
    </div>
  </div>
</nav>
<script>
function decodeJWT(token){
  try{ return JSON.parse(atob(token.split('.')[1])); }catch(e){ return {}; }
}
function updateNavbar(){
  const token = localStorage.getItem('jwt');
  const navUser = document.getElementById('navUser');
  if(token){
    const data = decodeJWT(token);
    const name = data.username || 'User';
    navUser.innerHTML = `<span class='me-2 fw-semibold'>👋 Hi, ${name}</span>
      <a class='btn btn-outline-danger btn-sm' href='/logout' onclick='localStorage.removeItem("jwt")'>Logout</a>`;
  }
}
document.addEventListener('DOMContentLoaded', updateNavbar);
</script>
<div class="container my-4">
"""

BASE_FOOT = """
</div>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
</body></html>
"""

# -------------------- FRONT PAGE --------------------
FRONT_HTML = BASE_HEAD + NAVBAR + """
<div class="hero">
  <h1 class="fw-bold display-4">KnowMap with AI</h1>
  <p class="lead text-muted mb-4">Visualize and explore knowledge intelligently.</p>
  <a href="/login" class="btn btn-primary btn-lg">Get Started</a>
</div>
""" + BASE_FOOT


# -------------------- LOGIN PAGE --------------------
LOGIN_HTML = BASE_HEAD + NAVBAR + """
<div class="login-card mx-auto mt-5 card p-4" style="max-width:400px;">
  <h4 class="text-center mb-3 fw-bold">Login</h4>
  <div id="loginForm">
    <input id="loginUsername" class="form-control mb-2" placeholder="Username">
    <input id="loginPassword" type="password" class="form-control mb-2" placeholder="Password">
    <button id="loginBtn" class="btn btn-primary w-100 mb-3">Login</button>
    <div class="text-center">No account? <a href="#" id="showSignup">Sign Up</a></div>
  </div>
  <div id="signupForm" style="display:none;">
    <input id="signupUsername" class="form-control mb-2" placeholder="Username">
    <input id="signupEmail" class="form-control mb-2" placeholder="Email">
    <input id="signupPassword" type="password" class="form-control mb-2" placeholder="Password">
    <button id="signupBtn" class="btn btn-success w-100 mb-3">Sign Up</button>
    <div class="text-center">Have account? <a href="#" id="showLogin">Login</a></div>
  </div>
</div>

<script>
document.getElementById('showSignup').onclick=()=>{loginForm.style.display='none';signupForm.style.display='block';};
document.getElementById('showLogin').onclick=()=>{signupForm.style.display='none';loginForm.style.display='block';};

document.getElementById('signupBtn').onclick=async()=>{
  const u=signupUsername.value.trim(), e=signupEmail.value.trim(), p=signupPassword.value;
  if(!u||!e||!p) return alert('Fill all fields');
  const r=await fetch('/signup',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({username:u,email:e,password:p})});
  if(r.ok){alert('Account created!'); showLogin.click();}
  else{alert('Signup failed');}
};

document.getElementById('loginBtn').onclick=async()=>{
  const u=loginUsername.value.trim(), p=loginPassword.value;
  const r=await fetch('/login',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({username:u,password:p})});
  const j=await r.json();
  if(r.ok){localStorage.setItem('jwt',j.token);window.location='/upload';}
  else alert('Invalid credentials');
};
</script>
""" + BASE_FOOT


# -------------------- UPLOAD PAGE --------------------
UPLOAD_HTML = BASE_HEAD + NAVBAR + """
<h4 class="fw-bold mb-3">Upload CSV or Analyze Sentence</h4>
<div class="row g-3">
  <div class="col-lg-6">
    <div class="card p-3">
      <h6>Upload CSV (subject,relation,object)</h6>
      <input type="file" id="csvFile" class="form-control mb-2" accept=".csv">
      <button id="uploadBtn" class="btn btn-primary w-100">Upload</button>
      <div id="uploadStatus" class="small mt-2 text-muted"></div>
    </div>
    <div class="card p-3 mt-3">
      <h6>Manual sentence</h6>
      <textarea id="sentence" rows="4" class="form-control">Albert Einstein developed the theory of relativity in 1905.</textarea>
      <button id="analyzeBtn" class="btn btn-success w-100 mt-2">Analyze</button>
    </div>
  </div>
  <div class="col-lg-6">
    <div class="card p-3 mb-3">
      <h6>Graph Preview</h6>
      <div id="graph"></div>
    </div>
    <div class="card p-3">
      <h6>Stats</h6>
      <div>Files uploaded: <span id="filesUploaded">0</span></div>
      <div>Triples total: <span id="triplesTotal">0</span></div>
      <div>Unique nodes: <span id="uniqueNodes">0</span></div>
    </div>
  </div>
</div>

<script src="https://unpkg.com/vis-network@9.1.2/dist/vis-network.min.js"></script>
<script>
function drawGraph(triples){
  const c=document.getElementById('graph');
  if(!triples||!triples.length){c.innerHTML='<div class="text-muted p-3">No data</div>';return;}
  const nodes=[],edges=[],ids=new Set();
  triples.forEach(t=>{
    const s=t.subject,o=t.object;
    if(!ids.has(s)){nodes.push({id:s,label:s});ids.add(s);}
    if(!ids.has(o)){nodes.push({id:o,label:o});ids.add(o);}
    edges.push({from:s,to:o,label:t.relation});
  });
  new vis.Network(c,{nodes:new vis.DataSet(nodes),edges:new vis.DataSet(edges)},{nodes:{shape:'box'}});
}
uploadBtn.onclick=async()=>{
  const f=csvFile.files[0];
  if(!f)return alert('Select file');
  const fd=new FormData();fd.append('file',f);
  const r=await fetch('/upload',{method:'POST',body:fd});
  const j=await r.json();
  if(r.ok){uploadStatus.textContent='Uploaded '+j.triples_added+' triples';drawGraph(j.triples);}
  else alert(j.error);
  refreshStats();
};
analyzeBtn.onclick=async()=>{
  const t=sentence.value.trim();
  const r=await fetch('/analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:t})});
  const j=await r.json();drawGraph(j.triples);refreshStats();
};
async function refreshStats(){
  const r=await fetch('/stats');const s=await r.json();
  filesUploaded.textContent=s.files_uploaded;
  triplesTotal.textContent=s.triples_total;
  uniqueNodes.textContent=s.unique_nodes;
}
refreshStats();
</script>
""" + BASE_FOOT


# -------------------- SEMANTIC SEARCH PAGE --------------------
SEMANTIC_HTML = BASE_HEAD + NAVBAR + """
<h4 class="fw-bold mb-3">Semantic Search</h4>
<div class="row">
  <div class="col-lg-8">
    <div class="card p-3">
      <input id="query" class="form-control mb-2" placeholder="Enter query">
      <button id="searchBtn" class="btn btn-primary w-100 mb-2">Search</button>
      <div id="graph"></div>
    </div>
  </div>
  <div class="col-lg-4">
    <div class="card p-3">
      <h6>Stats</h6>
      <div>Files uploaded: <span id="filesUploaded">0</span></div>
      <div>Triples total: <span id="triplesTotal">0</span></div>
      <div>Unique nodes: <span id="uniqueNodes">0</span></div>
    </div>
  </div>
</div>
<script src="https://unpkg.com/vis-network@9.1.2/dist/vis-network.min.js"></script>
<script>
function drawGraph(triples){
  const c=document.getElementById('graph');
  if(!triples||!triples.length){c.innerHTML='<div class="text-muted p-3">No data</div>';return;}
  const nodes=[],edges=[],ids=new Set();
  triples.forEach(t=>{
    if(!ids.has(t.subject)){nodes.push({id:t.subject,label:t.subject});ids.add(t.subject);}
    if(!ids.has(t.object)){nodes.push({id:t.object,label:t.object});ids.add(t.object);}
    edges.push({from:t.subject,to:t.object,label:t.relation});
  });
  new vis.Network(c,{nodes:new vis.DataSet(nodes),edges:new vis.DataSet(edges)},{nodes:{shape:'box'}});
}
searchBtn.onclick=async()=>{
  const q=query.value.trim();
  const r=await fetch('/search',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query:q})});
  const j=await r.json();drawGraph(j.triples);refreshStats();
};
async function refreshStats(){
  const r=await fetch('/stats');const s=await r.json();
  filesUploaded.textContent=s.files_uploaded;
  triplesTotal.textContent=s.triples_total;
  uniqueNodes.textContent=s.unique_nodes;
}
refreshStats();
</script>
""" + BASE_FOOT


# -------------------- DASHBOARD PAGE --------------------
DASHBOARD_HTML = BASE_HEAD + NAVBAR + """
<div class="dashboard-scroll">
<h4 class="fw-bold mb-3">Dashboard</h4>
<div class="row g-3">
  <div class="col-md-4"><div class="card p-3"><h6>Files uploaded</h6><h4 id="filesUploaded">0</h4></div></div>
  <div class="col-md-4"><div class="card p-3"><h6>Triples total</h6><h4 id="triplesTotal">0</h4></div></div>
  <div class="col-md-4"><div class="card p-3"><h6>Unique nodes</h6><h4 id="uniqueNodes">0</h4></div></div>
</div>
<div class="card p-3 my-3"><h6>Weekly Usage</h6><canvas id="usageChart"></canvas></div>
<div class="card p-3">
  <h6>Feedback</h6>
  <div id="stars">
    <span class="star" data-v="1">&#9733;</span><span class="star" data-v="2">&#9733;</span><span class="star" data-v="3">&#9733;</span><span class="star" data-v="4">&#9733;</span><span class="star" data-v="5">&#9733;</span>
  </div>
  <textarea id="manual" class="form-control mb-2" placeholder="Write feedback..."></textarea>
  <button id="send" class="btn btn-outline-primary w-100">Submit</button>
  <div class="mt-2 text-muted small">Average rating: <span id="avgRating">0</span></div>
  <hr><h6>Recent Feedback</h6><div id="feedbackList" style="max-height:200px;overflow:auto;"></div>
</div></div>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script>
let rating=0;
document.querySelectorAll('.star').forEach(s=>s.onclick=()=>{rating=parseInt(s.dataset.v);document.querySelectorAll('.star').forEach(x=>x.classList.toggle('selected',parseInt(x.dataset.v)<=rating));});
send.onclick=async()=>{
  if(!rating)return alert('Select rating');
  const m=manual.value.trim();
  const r=await fetch('/feedback',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({rating,manual:m})});
  if(r.ok){manual.value='';rating=0;document.querySelectorAll('.star').forEach(x=>x.classList.remove('selected'));loadDashboard();}
};
async function loadDashboard(){
  const s=await (await fetch('/stats')).json();
  filesUploaded.textContent=s.files_uploaded;triplesTotal.textContent=s.triples_total;uniqueNodes.textContent=s.unique_nodes;avgRating.textContent=s.avg_rating||0;
  const f=await (await fetch('/get_feedback')).json();
  feedbackList.innerHTML='';f.feedback.slice(-10).reverse().forEach(i=>{feedbackList.innerHTML+=`<div class='border p-2 mb-2 rounded'><b>${i.username}</b> - ${i.rating}/5<br>${i.manual||''}</div>`;});
  const m=await (await fetch('/metrics')).json();
  const ctx=document.getElementById('usageChart');if(window._chart){window._chart.data.labels=m.labels;window._chart.data.datasets[0].data=m.counts;window._chart.update();}
  else window._chart=new Chart(ctx,{type:'bar',data:{labels:m.labels,datasets:[{label:'Usage',data:m.counts,backgroundColor:'rgba(54,162,235,0.6)'}]},options:{scales:{y:{beginAtZero:true}}}});
}
setInterval(loadDashboard,3000);loadDashboard();
</script>
""" + BASE_FOOT


# -------------------- ROUTES --------------------
@app.route("/")
def front(): return render_template_string(FRONT_HTML)

@app.route("/login", methods=["GET","POST"])
def login_page():
    if request.method=="GET": return render_template_string(LOGIN_HTML)
    data=request.get_json();u,p=data.get("username",""),data.get("password","")
    conn=sqlite3.connect(DB);c=conn.cursor();c.execute("SELECT password_hash FROM users WHERE username=?",(u,));row=c.fetchone();conn.close()
    if not row or not check_password_hash(row[0],p): return jsonify({"error":"Invalid"}),401
    token=jwt.encode({"username":u,"ts":time.time()},SECRET_KEY,algorithm=JWT_ALGO)
    return jsonify({"token":token})

@app.route("/signup", methods=["POST"])
def signup():
    d=request.get_json();u=d["username"].strip();e=d["email"].strip();p=d["password"]
    conn=sqlite3.connect(DB);c=conn.cursor()
    try:
        c.execute("INSERT INTO users(username,email,password_hash)VALUES(?,?,?)",(u,e,generate_password_hash(p)));conn.commit()
    except sqlite3.IntegrityError: return jsonify({"error":"exists"}),400
    conn.close();return jsonify({"ok":1})

@app.route("/logout")
def logout(): return redirect(url_for("login_page"))

@app.route("/upload", methods=["GET","POST"])
def upload_page():
    if request.method=="GET": return render_template_string(UPLOAD_HTML)
    f=request.files.get("file")
    if not f: return jsonify({"error":"no file"}),400
    reader=csv.reader(io.StringIO(f.stream.read().decode()))
    added=[]
    for r in reader:
        if len(r)<3: continue
        s,rn,o=r[0].strip(),r[1].strip(),r[2].strip()
        t={"subject":s,"relation":rn,"object":o}
        if t not in semantic_search_data: semantic_search_data.append(t);added.append(t)
    STATS["files_uploaded"]+=1;recalc_node_counts();bump_processing(len(added))
    return jsonify({"triples_added":len(added),"triples":added})

@app.route("/semantic")
def semantic_page(): return render_template_string(SEMANTIC_HTML)

@app.route("/dashboard")
def dashboard_page(): return render_template_string(DASHBOARD_HTML)

@app.route("/analyze", methods=["POST"])
def analyze():
    t=request.get_json().get("text","");doc=nlp(t);trip=[]
    for s in doc.sents:
        r=s.root.text
        for n in s.noun_chunks:
            trip.append({"subject":n.text,"relation":"related_to","object":r})
    semantic_search_data.extend(trip);recalc_node_counts();bump_processing()
    return jsonify({"triples":trip})

@app.route("/search", methods=["POST"])
def search():
    q=request.get_json().get("query","")
    nodes=list({t["subject"] for t in semantic_search_data}|{t["object"] for t in semantic_search_data})
    if not nodes:return jsonify({"triples":[]})
    emb=embed_model.encode(nodes,convert_to_numpy=True)
    qv=embed_model.encode([q],convert_to_numpy=True)
    sc=cosine_similarity(qv,emb)[0];idx=np.argsort(sc)[::-1][:5]
    top=[{"subject":nodes[i],"relation":"related_to","object":q} for i in idx]
    return jsonify({"triples":top})

@app.route("/stats")
def stats(): STATS["avg_rating"]=compute_avg_rating();return jsonify(STATS)

@app.route("/metrics")
def metrics(): return jsonify({"labels":["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],"counts":[PROCESSING_DAILY.get(i,0) for i in range(7)]})

@app.route("/feedback", methods=["POST"])
def feedback():
    d=request.get_json();r=int(d.get("rating",0));m=d.get("manual","")
    FEEDBACK_LIST.append({"username":"user","rating":r,"manual":m})
    return jsonify({"ok":1})

@app.route("/get_feedback")
def get_feedback(): return jsonify({"feedback":FEEDBACK_LIST})


if __name__ == "__main__":
    print("🚀 Running on http://127.0.0.1:5000")
    app.run(debug=True)
