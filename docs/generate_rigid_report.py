import argparse
import json
from pathlib import Path

import numpy as np


def main():
  p = argparse.ArgumentParser()
  p.add_argument("data", type=Path)
  p.add_argument("--output", type=Path, required=True)
  p.add_argument("--video", default="")
  a = p.parse_args()
  d = np.load(a.data, allow_pickle=True)
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
  D["joint_names"] = n
  dur = float(d.get("measurement_duration", 10))
  e = float(d.get("total_energy", 0))
  dist = float(d.get("total_distance", 0))
  m = 5.424414725
  h = """<!doctype html><html><head><meta charset="utf-8"><script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script><style>body{font:14px Arial;background:#eef2f4;margin:0}main{max-width:1400px;margin:auto;padding:20px}.top,.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}.card{background:#fff;padding:18px;margin:10px 0;border-radius:14px;box-shadow:0 2px 10px #0001}.plot{height:330px;background:#eef2f4;border:1px solid #cbd5dc;border-radius:14px;overflow:hidden}.wide{grid-column:1/-1}button{padding:9px;margin:3px;border:0;border-radius:6px}button.active{background:#1769aa;color:#fff}.panel{display:none}.panel.active{display:block}video{width:100%;max-height:500px}</style></head><body><main><div class="top"><section class="card"><h1>Робот-гепард — фиксированная спина <span style="float:right;color:#1769aa">ИТМО</span></h1>VIDEO</section><section class="card"><h2>Характеристики</h2><p>Масса: MASS кг<br>Скорость: SPEED м/с<br>Энергия: ENERGY Дж<br>Время: DUR с<br>Дистанция: DIST м<br>CoT: COT</p></section></div><section class="card"><details><summary><h2 style="display:inline">Actuator metadata</h2></summary><p>CubeMars AK40-10 и AK45-10; характеристики приводов указаны в документации.</p></details></section><section class="card"><h2>Simulation data</h2><input id="slider" type="range" min="0" max="MAX" style="width:100%"><span id="clock">0 s</span><div id="buttons"></div><div id="panels"></div></section></main><script>const D=DATA,T={};const B=document.getElementById('buttons'),P=document.getElementById('panels');function L(t,x,y){return{title:{text:t,x:.03,xanchor:'left',font:{family:'DejaVu Sans Mono',size:13,color:'#1f2933'}},paper_bgcolor:'#eef2f4',plot_bgcolor:'#eef2f4',xaxis:{title:x},yaxis:{title:y},margin:{t:55,r:20,b:55,l:65}}}function add(n,k,y){let b=document.createElement('button'),s=document.createElement('div');b.textContent=n;b.onclick=()=>{Object.values(T).forEach(x=>x.className='panel');s.className='panel active';B.querySelectorAll('button').forEach(x=>x.classList.toggle('active',x==b))};B.append(b);s.className='panel';T[n]=s;P.append(s);if(k=='energy'){s.innerHTML='<div id="energy" class="plot wide"></div>';Plotly.newPlot('energy',[{x:D.time,y:D.energy_trace}],L('Total electrical energy','Time [s]','Energy [J]'));return}if(k=='com'){s.innerHTML='<div class="grid"><div id="cp" class="plot"></div><div id="cv" class="plot"></div></div>';Plotly.newPlot('cp',['x','y','z'].map((q,i)=>({x:D.time,y:D.com_position.map(r=>r[i]),name:q})),L('COM — position','Time [s]','Position [m]'));Plotly.newPlot('cv',['x','y','z'].map((q,i)=>({x:D.time,y:D.com_velocity.map(r=>r[i]),name:q})),L('COM — derivative','Time [s]','Velocity [m/s]'));return}s.innerHTML='<div class="grid">'+D.joint_names.map((_,i)=>'<div id="'+k+i+'" class="plot"></div>').join('')+'</div>';D.joint_names.forEach((j,i)=>Plotly.newPlot(k+i,[{x:D.time,y:D[k].map(r=>r[i]),name:j}],L(n=='Phase portraits'?j:j,'Time [s]',y)))}[['COM','com',''],['Positions','position','Position [rad]'],['Velocities','velocity','Velocity [rad/s]'],['Torques','torque','Torque [N·m]'],['Phase portraits','velocity','Velocity [rad/s]'],['Mechanical power','mechanical_power','Power [W]'],['Electric power','power','Power [W]'],['Energy','energy','Energy [J]']].forEach(x=>add(...x));document.querySelector('button').click();slider.oninput=()=>{clock.textContent=D.time[slider.value].toFixed(3)+' s';let v=document.querySelector('video');if(v)v.currentTime=D.time[slider.value]};</script></body></html>"""
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
  a.output.write_text(h, encoding="utf-8")


if __name__ == "__main__":
  main()
