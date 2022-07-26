const openModalBtn = document.getElementById('addPatient');
const closeModalBtn = document.getElementById('patientModalClose');
const modal = document.getElementById('patientModal');
const modalWrapper = document.getElementById('patientModalWrapper');
const modalContent = document.getElementById('patientModalContent');

function addShowHideModal() {
  openModalBtn.onclick = function(event) {
    modal.style.display = "block";
    //console.log("Modal shown");
  }

  window.onclick = function(event) {
    // Close the modal if the backdrop is clicked
    // Excluding button prevents an open and immediate close when clicking 
    target = event.target;
    //console.log(event.target);
    if (
      target == modal | target == closeModalBtn | target == modalWrapper && 
      target != openModalBtn && modal.style.display != "none"
    ) {
      modal.style.display = "none";
      //console.log("Modal hidden");
    }
  }
}
