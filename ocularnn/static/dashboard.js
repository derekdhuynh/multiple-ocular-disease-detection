const btn = document.getElementById('addPatient');
const modal = document.getElementById('hello');

function addShowHideModal() {
  btn.onclick = function(event) {
    modal.hidden = false;
    console.log("Modal shown");
  }

  window.onclick = function(event) {
    if (event.target != modal && event.target != btn && modal.hidden == false) {
      modal.hidden = true;
      console.log("Modal hidden");
    }
  }
}
