async function recordAndPlay() {
    const stream = await navigator.mediaDevices.getUserMedia({audio:true});
    const recorder = new MediaRecorder(stream,{mimeType:'audio/webm'});
    let data=[];
    recorder.ondataavailable = e=>data.push(e.data);
    recorder.start(); setTimeout(()=>recorder.stop(),1500);
    recorder.onstop = async ()=>{
      const blob = new Blob(data,{type:'audio/webm'});
      const buf  = await blob.arrayBuffer();
      const res  = await fetch('/play',{method:'POST',body:buf});
      const js   = await res.json();
      if(js.error==='wait'){alert('Wait');return;}
      document.getElementById('board').innerHTML = js.board_html;
      document.getElementById('msg').innerText = js.message;
    }
  }
  document.getElementById('recBtn').onclick = recordAndPlay;
  