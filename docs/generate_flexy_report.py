"""Generate an interactive report for the flexible-spine Cheetah."""

import argparse
import json
from html import escape
from pathlib import Path

import numpy as np


def main():
  p = argparse.ArgumentParser()
  p.add_argument("data", type=Path)
  p.add_argument("--output", type=Path, required=True)
  p.add_argument("--video", default="")
  p.add_argument("--training-path", default="")
  a = p.parse_args()
  d = np.load(a.data, allow_pickle=True)
  training_path = a.training_path or str(d.get("training_path", ""))
  n = d["joint_names"].astype(str).tolist()
  D = {
    k: d[k].tolist()
    for k in (
      "time",
      "position",
      "velocity",
      "torque",
      "mechanical_power",
      "power",
      "energy_trace",
      "com_position",
      "com_velocity",
    )
    if k in d
  }
  ak40_specs = (
    ("Масса", "190 г"),
    ("Максимальная длительная мощность", "50 Вт"),
    ("Номинальное напряжение", "24 В"),
    ("Номинальный момент", "1.3 Н·м"),
    ("Пиковый момент", "4.1 Н·м"),
    ("Номинальная скорость", "370 об/мин"),
    ("Скорость холостого хода", "435 об/мин"),
    ("Номинальный ток", "2.7 А"),
    ("Пиковый ток", "7.3 А"),
    ("Максимальная удельная величина момента", "20.5 Н·м/кг"),
    ("Базовая динамическая грузоподъёмность C", "2810 Н"),
    ("Базовая статическая грузоподъёмность C₀", "2760 Н"),
    ("Уровень шума на расстоянии 65 см", "50 дБ"),
    ("Момент обратного проворачивания", "0.06 Н·м"),
    ("Люфт", "18 угл. мин"),
    ("Инерция ротора", "97.35 г·см²"),
    ("Постоянная двигателя", "0.0568 Н·м/√Вт"),
    ("Постоянная скорости", "170 об/мин/В"),
    ("Постоянная момента", "0.056 Н·м/А"),
    ("Постоянная противо-ЭДС", "5.88 В/коб/мин"),
    ("Фазное сопротивление", "978 мОм"),
    ("Фазная индуктивность", "465 мкГн"),
  )
  ak45_values = (
    "262 г",
    "39 Вт",
    "24 В",
    "2.5 Н·м",
    "7 Н·м",
    "120 об/мин",
    "180 об/мин",
    "1.9 А",
    "5 А",
    "26.9 Н·м/кг",
    "2810 Н",
    "2760 Н",
    "50 дБ",
    "0.1 Н·м",
    "18 угл. мин",
    "157.33 г·см²",
    "0.0858 Н·м/√Вт",
    "75 об/мин/В",
    "0.127 Н·м/А",
    "13.33 В/коб/мин",
    "2200 мОм",
    "1330 мкГн",
  )
  rows = []
  for name in n:
    is_knee = "knee_pitch" in name
    motor = "AK40-10" if is_knee else "AK45-10"
    state = "Активный"
    values = tuple(value for _, value in ak40_specs) if is_knee else ak45_values
    cells = (name, "CubeMars " + motor, state, *values)
    rows.append("<tr>" + "".join("<td>" + escape(v) + "</td>" for v in cells) + "</tr>")
  headers = ("Сустав", "Привод", "Состояние", *(label for label, _ in ak40_specs))
  actuator_table = (
    '<p style="color:#717b85;font-size:13px">Паспортные характеристики '
    "привода каждого сустава. Прокрутите таблицу вправо для просмотра "
    "всех параметров.</p>"
    '<div class="actuator-table"><table><thead><tr>'
    + "".join("<th>" + escape(label) + "</th>" for label in headers)
    + "</tr></thead><tbody>"
    + "".join(rows)
    + "</tbody></table></div>"
  )
  D["joint_names"] = n
  dur = float(d.get("measurement_duration", 10))
  e = float(d.get("total_energy", 0))
  dist = float(d.get("total_distance", 0))
  m = 5.424414725
  h = """<!doctype html><html><head><meta charset="utf-8"><link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin><link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@500&display=swap" rel="stylesheet"><script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script><style>body{font:500 14px 'Montserrat',sans-serif;background:#e2e7eb;margin:0}main{max-width:1400px;margin:auto;padding:20px}.top,.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}.card{background:#fff;border:1px solid #cbd5dc;padding:18px;margin:10px 0;border-radius:14px;box-shadow:0 2px 10px #0001}.plot{height:330px;background:#eef2f4;border:1px solid #cbd5dc;border-radius:14px;overflow:hidden}.wide{grid-column:1/-1}button{padding:9px;margin:3px;border:0;border-radius:6px}button.active{background:#1769aa;color:#fff}.panel{display:none}.panel.active{display:block}video{width:100%;max-height:500px}.metrics-grid{display:grid;grid-template-columns:repeat(3,1fr);width:max-content;max-width:100%;gap:10px}
.metric{position:relative;box-sizing:border-box;padding:36px 14px 18px;background:#eef2f4;border:1px solid #cbd5dc;border-radius:14px;display:flex;align-items:center;min-width:0}
.metric-label{position:absolute;top:12px;left:14px;color:#717b85;font-size:13px}
.metric-value{white-space:nowrap;font-size:clamp(16px,1.6vw,24px);font-weight:600;color:#1f2933;font-variant-numeric:tabular-nums}
@media(max-width:700px){.top{grid-template-columns:1fr}.metrics-grid{grid-template-columns:repeat(2,1fr)}}
h1,h2,button,input,summary,.metric-value{font-family:'Montserrat',sans-serif;font-weight:500}button,input{font-size:inherit}.actuator-table{overflow-x:auto;margin-top:16px;border:1px solid #cbd5dc;border-radius:14px;background:#eef2f4}.actuator-table table{width:100%;border-collapse:collapse;text-align:left;font-size:13px}.actuator-table th,.actuator-table td{padding:12px 14px;border-bottom:1px solid #cbd5dc}.actuator-table td{white-space:nowrap}.actuator-table th{min-width:120px;color:#717b85;font-weight:500}.actuator-table tr:last-child td{border-bottom:0}#energy{width:80%;margin:0 auto;box-sizing:border-box}</style></head><body><main><div class="top"><section class="card"><h1>Робот-гепард — гибкая спина <span style="float:right;color:#1769aa">ИТМО</span></h1>VIDEO</section><section class="card"><h2>Характеристики</h2><div class="metrics-grid"><div class="metric"><span class="metric-label">Масса</span><span class="metric-value">MASS кг</span></div><div class="metric"><span class="metric-label">Скорость</span><span class="metric-value">SPEED м/с</span></div><div class="metric"><span class="metric-label">Энергия</span><span class="metric-value">ENERGY Дж</span></div><div class="metric"><span class="metric-label">Время</span><span class="metric-value">DUR с</span></div><div class="metric"><span class="metric-label">Дистанция</span><span class="metric-value">DIST м</span></div><div class="metric"><span class="metric-label">CoT</span><span class="metric-value">COT</span></div></div>TRAINING_PATH</section></div><section class="card"><details><summary><h2 style="display:inline">Actuator metadata</h2></summary>ACTUATOR_TABLE</details></section><section class="card"><h2>Simulation data</h2><input id="slider" type="range" min="0" max="MAX" style="width:100%"><span id="clock">0 s</span><div id="buttons"></div><div id="panels"></div></section></main><script>const D=DATA,T={};const B=document.getElementById('buttons'),P=document.getElementById('panels');function L(t,x,y){return{title:{text:t,x:.03,xanchor:'left',font:{family:'DejaVu Sans Mono',size:13,color:'#1f2933'}},paper_bgcolor:'#eef2f4',plot_bgcolor:'#eef2f4',xaxis:{title:x},yaxis:{title:y},shapes:[{type:'line',xref:'x',yref:'paper',x0:D.time[0],x1:D.time[0],y0:0,y1:1,line:{color:'#d1495b',width:2}}],margin:{t:55,r:20,b:55,l:65}}}function add(n,k,y){let b=document.createElement('button'),s=document.createElement('div');b.textContent=n;b.onclick=()=>{Object.values(T).forEach(x=>x.className='panel');s.className='panel active';B.querySelectorAll('button').forEach(x=>x.classList.toggle('active',x==b));requestAnimationFrame(()=>s.querySelectorAll('.plot').forEach(plot=>{if(plot.layout)Plotly.Plots.resize(plot)}))};B.append(b);s.className='panel';T[n]=s;P.append(s);if(k=='energy'){s.innerHTML='<div id="energy" class="plot wide"></div>';Plotly.newPlot('energy',[{x:D.time,y:D.energy_trace}],L('Total electrical energy','Time [s]','Energy [J]'));return}if(k=='com'){s.innerHTML='<div class="grid"><div id="cp" class="plot"></div><div id="cv" class="plot"></div></div>';Plotly.newPlot('cp',['x','y','z'].map((q,i)=>({x:D.time,y:D.com_position.map(r=>r[i]),name:q})),L('COM — position','Time [s]','Position [m]'));Plotly.newPlot('cv',['x','y','z'].map((q,i)=>({x:D.time,y:D.com_velocity.map(r=>r[i]),name:q})),L('COM — derivative','Time [s]','Velocity [m/s]'));return}s.innerHTML='<div class="grid">'+D.joint_names.map((_,i)=>'<div id="'+n+k+i+'" class="plot"></div>').join('')+'</div>';D.joint_names.forEach((j,i)=>{
const phase=n==='Phase portraits';
const layout=L(j,phase?'Момент [Н·м]':'Time [s]',phase?'Скорость [рад/с]':y);
const traces=[{x:phase?D.torque.map(r=>r[i]):D.time,
y:D[k].map(r=>r[i]),name:j,mode:phase?'markers':'lines',marker:{size:4}}];
const plot=document.getElementById(n+k+i);
if(phase){layout.shapes=[];plot.dataset.jointIndex=i;
traces.push({x:[D.torque[0][i]],y:[D.velocity[0][i]],mode:'markers',
name:'Текущее время',marker:{size:10,color:'#d1495b',line:{color:'#fff',width:1}}})}
Plotly.newPlot(plot,traces,layout)
})}[['COM','com',''],['Positions','position','Position [rad]'],['Velocities','velocity','Velocity [rad/s]'],['Torques','torque','Torque [N·m]'],['Phase portraits','velocity','Velocity [rad/s]'],['Mechanical power','mechanical_power','Power [W]'],['Electric power','power','Power [W]'],['Energy','energy','Energy [J]']].forEach(x=>add(...x));document.querySelector('button').click();const slider=document.getElementById('slider');
const clock=document.getElementById('clock');
const video=document.querySelector('video');
slider.value=0;
let cursorFrame=null;
function displayTime(i){
slider.value=i;clock.textContent=D.time[i].toFixed(3)+' s';
if(cursorFrame!==null)cancelAnimationFrame(cursorFrame);
cursorFrame=requestAnimationFrame(()=>{
cursorFrame=null;
const time=D.time[Number(slider.value)];
document.querySelectorAll('.plot').forEach(plot=>{
if(!plot.layout)return;
if(plot.dataset.jointIndex!==undefined){
const joint=Number(plot.dataset.jointIndex),sample=Number(slider.value);
Plotly.restyle(plot,{x:[[D.torque[sample][joint]]],
y:[[D.velocity[sample][joint]]]},[1]);
}else Plotly.relayout(plot,{'shapes[0].x0':time,'shapes[0].x1':time});
});
});
}
slider.oninput=()=>{const i=Number(slider.value);displayTime(i);
if(video){video.pause();video.currentTime=D.time[i]-D.time[0]}};
if(video){video.addEventListener('timeupdate',()=>{
const t=video.currentTime+D.time[0];let lo=0,hi=D.time.length-1;
while(lo<hi){const mid=Math.ceil((lo+hi)/2);
if(D.time[mid]<=t)lo=mid;else hi=mid-1}displayTime(lo)});
video.addEventListener('loadedmetadata',()=>{video.currentTime=0;displayTime(0)})}
</script></body></html>"""
  h = (
    h.replace(
      "VIDEO",
      f'<video controls src="{a.video}"></video>' if a.video else "Видео не указано",
    )
    .replace("DATA", json.dumps(D))
    .replace("MAX", str(len(D["time"]) - 1))
    .replace("MASS", f"{m:.3f}")
    .replace("SPEED", f"{dist / dur:.3f}")
    .replace("ENERGY", f"{e:.2f}")
    .replace("DUR", f"{dur:.2f}")
    .replace("DIST", f"{dist:.2f}")
    .replace("COT", f"{e / (m * 9.81 * dist):.5f}" if dist else "n/a")
  )
  h = h.replace(
    "TRAINING_PATH",
    '<p style="color:#717b85;font-size:13px;overflow-wrap:anywhere;'
    'margin:18px 0 0">Обучение: ' + escape(training_path) + "</p>"
    if training_path
    else "",
  )
  h = h.replace("ACTUATOR_TABLE", actuator_table)
  a.output.write_text(h, encoding="utf-8")


if __name__ == "__main__":
  main()
