using UnityEngine;

namespace UserMovementRecording
{
    public class TrackGameObjects : MonoBehaviour
    {
        public GameObject leftHand;
    
        public GameObject rightHand;
    
        public GameObject hmd;

        public bool isValid = false;
        
        private void Start()
        {
            
            if (leftHand.activeSelf && rightHand.activeSelf && hmd.activeSelf){
                isValid = true;
                Debug.Log("left hand: " + leftHand.activeSelf);
                Debug.Log("right hand: " + rightHand.activeSelf);
                Debug.Log("hmd: " + hmd.activeSelf);
            }
            else
            {
                isValid = false;
            }

        }

    }
}