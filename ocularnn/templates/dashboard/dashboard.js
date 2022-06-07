const btn = document.getElementById('addPatient');
const modal = document.getElementById('hello');

console.log(modal.hidden);
console.log("Hello world");
alert("Hello world");

function showHideModal(isHidden) {
  if (isHidden) {
    modal.hidden = true;
  }
  if (!isHidden) {
    modal.hidden = false;
  }
}

btn.onclick = function() {
  showHideModal(modal.hidden);
}

window.onclick = function() {
  showHideModal(modal.hidden);
}

