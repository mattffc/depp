let d = new Date();
//document.body.innerHTML = "<h1>Time right now is:  " + d.getHours() + ":" + d.getMinutes() + ":" + d.getSeconds()</h1>"
console.log("loool234")



const handleImageUpload = event => {
    // here need to change a variable in the html, the graph values
    
  const files = event.target.files
  const formData = new FormData()
  formData.append('myFile', files[0])

  fetch('/saveImage', {
    method: 'POST',
    body: formData
  })
  .then(response => response.json())
  .then(data => {
    console.log(data.path)
  })
  .catch(error => {
    console.error(error)
  })
  console.log("about to do stuff")
  
  
}

document.querySelector('#fileUpload').addEventListener('change', event => {
  handleImageUpload(event)
})

