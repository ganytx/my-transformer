const clamp=x=>Math.max(0,Math.min(1,x));
const ease=x=>1-Math.pow(1-clamp(x),3);
const appear=(sel,t,delay=0,dy=45)=>{const e=document.querySelector(sel);if(!e)return;const p=ease((t-delay)/.65);e.style.opacity=p;e.style.transform=`translateY(${(1-p)*dy}px)`};
const duration=Number(document.body.dataset.duration);
const scene=document.body.dataset.scene;
window.film={duration,frame(t){
 document.getElementById('grid').style.transform=`translate(${t*3}px,${t*1.5}px)`;
 document.querySelector('.line').style.width=`${clamp(t/duration)*100}%`;
 document.querySelector('.glow').style.transform=`translateX(${Math.sin(t*.6)*40}px)`;
 const fade=clamp((duration-t)/.22);document.querySelector('.content').style.opacity=fade;
 if(scene==='hook'){appear('.label',t,.1);appear('.headline',t,.25,70);appear('.sub',t,.65);appear('.hero-terminal',t,1);document.querySelector('#typed').textContent='flash'.slice(0,Math.floor(clamp((t-1.15)/.7)*5));}
 if(scene==='init'){appear('.label',t,0);appear('.headline',t,.12);appear('.terminal',t,.25);const cmd='flash init my-next-idea';document.querySelector('#typed').textContent=cmd.slice(0,Math.floor(clamp((t-.5)/1.15)*cmd.length));appear('#output1',t,1.8,12);appear('#output2',t,2.2,12);appear('#output3',t,2.6,12);}
 if(scene==='features'){appear('.label',t,0);appear('.headline',t,.12);['#c1','#c2','#c3'].forEach((s,i)=>appear(s,t,.4+i*.22));}
 if(scene==='flow'){appear('.label',t,0);appear('.headline',t,.1);appear('.sub',t,.35);['#cmd1','#cmd2','#cmd3'].forEach((s,i)=>appear(s,t,.65+i*.7,30));appear('#done',t,3.1);}
 if(scene==='end'){appear('.chip',t,.1);appear('.wordmark',t,.18,60);appear('.endline',t,.5);appear('.sub',t,.8);appear('.cta',t,1.1);}
 const cursor=document.querySelector('.cursor');if(cursor)cursor.style.opacity=Math.floor(t*2)%2?0:1;
}};
