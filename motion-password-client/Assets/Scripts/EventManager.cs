using System.Threading.Tasks;
using UnityEngine;

public class EventManager : MonoBehaviour
{
    public StateManager stateManager;

    // Start is called before the first frame update
    void Start()
    {
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    public void TabEnrollButtonClicked()
    {
        stateManager.SwitchState();
    }
    
    public void TabVerifyButtonClicked()
    {
        stateManager.SwitchState();
    }
    
    public void NewUserButtonClicked()
    {
        stateManager.CreateNewUser();
    }
    
    public void UserButtonClicked(GameObject clickedGameObject)
    {
        stateManager.UserButtonClicked(clickedGameObject);
    }
    
    public void DeleteUserButtonClicked(GameObject clickedGameObject)
    {
        stateManager.DeleteUser(clickedGameObject);
    }
    
    public void StartStopButtonDown()
    {
        stateManager.ButtonDown();
    }
    
    public void ThresholdFirstValueButtonClicked()
    {
        stateManager.ThresholdClicked(1);
    }
    
    public void ThresholdSecondValueButtonClicked()
    {
        stateManager.ThresholdClicked(2);
    }
    
    public void ThresholdThirdValueButtonClicked()
    {
        stateManager.ThresholdClicked(3);
    }
    public void ThresholdFourthValueButtonClicked()
    {
        stateManager.ThresholdClicked(4);
    }

}