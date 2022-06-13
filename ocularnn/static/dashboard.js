const btn = document.getElementById('addPatient');
const modal = document.getElementById('patientModal');

function addShowHideModal() {
  btn.onclick = function(event) {
    modal.style.display = "block";
    console.log("Modal shown");
  }

  window.onclick = function(event) {
    if (event.target != modal && event.target != btn && modal.hidden == false) {
      modal.style.display = "none";
      console.log("Modal hidden");
    }
  }
}
