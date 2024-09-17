async function Analyze() {
    
    try {
      var image = document.getElementById('image').files[0];
      output.textContent = 'Analyzing...';
      const formdata = new FormData();
      formdata.append('file', image);
      formdata.append('name', image.name);
      const response = await fetch("http://127.0.0.1:8000/videoanalayzer", { method: 'POST', body: formdata });
      const message = await response.text();
      output.textContent = message.slice(1, -1);
    } catch (error) {
      output.textContent = `Error: ${error.message}`;
    }
  }
