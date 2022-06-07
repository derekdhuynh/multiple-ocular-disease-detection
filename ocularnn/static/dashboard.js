const btn = document.getElementById('addPatient');
const modal = document.getElementById('hello');

btn.onClick = function() {
  modal.hidden = false;
}

window.onClick = function() {
  modal.hidden = true;
}
