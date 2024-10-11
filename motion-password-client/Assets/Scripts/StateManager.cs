using System.Collections;
using System.Threading.Tasks;
using API;
using TMPro;
using UnityEngine;
using UserMovementRecording;

public class StateManager : MonoBehaviour
{
    public UIManager uIManager;
    public XRDataSaver XRDataSaver;
    
    public enum State { Verify, Enroll };
    
    private readonly State[] _states = { State.Enroll, State.Verify };
    private bool _recording;
    private int _currentStateIndex;
    private bool _cooldownWaited;
    private float _elapsedTime;
    private bool _isTimerRunning;
    private IEnumerator _stopwatchCoroutine;
    private User _activeUser;
    private GameObject _activeUserObject;
    private double _currentThreshold;
    
    public void Start()
    {
        _currentStateIndex = 0;
        _recording = false;
        _cooldownWaited = true;
        _activeUser = new User();
    }

    public void SwitchState()
    {
        if(_recording) return;
        _currentStateIndex = (_currentStateIndex + 1) % _states.Length;
        uIManager.UpdateUI(GetCurrentState());
    }

    public State GetCurrentState()
    {
        return _states[_currentStateIndex];
    }

    public void ButtonDown()
    {
        if (!_cooldownWaited) return;
        _recording = !_recording;
        if (_recording)
        {
            StartTimer();
            XRDataSaver.StartRecording();
        }
        else
        {
            StopTimer();
            XRDataSaver.StopRecording();
            uIManager.SendData();
        }
        StartCoroutine(WaitOneSecond());
    }
    
    IEnumerator WaitOneSecond()
    {
        _cooldownWaited = false;
        //uIManager.ChangeButtonAccessibility(_CooldownWaited);
        yield return new WaitForSeconds(2f);
        _cooldownWaited = true;
        //uIManager.ChangeButtonAccessibility(_CooldownWaited);
    }
    
    private void StartTimer()
    {
        if (_isTimerRunning) return;
        _isTimerRunning = true;
        _stopwatchCoroutine = StartStopwatch();
        StartCoroutine(_stopwatchCoroutine);
    }
    
    private IEnumerator StartStopwatch()
    {
        var startTime = Time.time;
        while (_isTimerRunning)
        {
            // Update elapsed time with seconds and milliseconds
            _elapsedTime = Time.time - startTime;

            // Calculate minutes, seconds, and milliseconds
            var minutes = Mathf.FloorToInt(_elapsedTime / 60f);
            var seconds = Mathf.FloorToInt(_elapsedTime % 60f);
            var milliseconds = Mathf.FloorToInt((_elapsedTime * 100f) % 100f);
            
            // Calculate color based on elapsed time
            var color = GetGradientColor(_elapsedTime);

            uIManager.ChangeTime(minutes, seconds, milliseconds, color);

            yield return new WaitForSecondsRealtime(.01f);
        }
    }
    private static Color GetGradientColor(float time)
    {
        switch (time)
        {
            case < 6f:
            {
                // Red to Yellow gradient (adjust values for desired color range)
                var t = Mathf.InverseLerp(0f, 6f, time);
                return Color.Lerp(Color.red, Color.yellow, t);
            }
            case < 10f:
            {
                // Yellow to Green gradient (adjust values for desired color range)
                var t = Mathf.InverseLerp(6f, 10f, time);
                return Color.Lerp(Color.yellow, Color.green, t);
            }
            default:
                return Color.green; // Green for times above 10 seconds
        }
    }

    private void StopTimer()
    {
        if (!_isTimerRunning) return;
        _isTimerRunning = false;
        StopCoroutine(_stopwatchCoroutine);
    }
    
    public async Task CreateNewUser()
    {
        if(_recording) return;
        var username = RandomUsernameGenerator.GetNewRandomUsername();
        await uIManager.CreateNewUser(username);
    }

    public void UserButtonClicked(GameObject clickedGameObject)
    {
        if(_recording) return;
        var button = clickedGameObject.transform.GetChild(0).gameObject;
        
        var textID = button.transform.GetChild(0).GetComponent<TMP_Text>(); 
        var textUsername = button.transform.GetChild(1).GetComponent<TMP_Text>();

        SetActiveUser(clickedGameObject, textID.text, textUsername.text);
    }

    public void SetActiveUser(GameObject clickedGameObject, string textIDText, string textUsernameText)
    {
        _activeUserObject = clickedGameObject;
        _activeUser.username = textUsernameText;
        _activeUser.id = int.Parse(textIDText);

        uIManager.WashoutHighlights();
        uIManager.HighlightActiveUser(clickedGameObject);
    }

    public User GetActiveUser()
    {
        return _activeUser;
    }
    
    public GameObject GetActiveUserObject()
    {
        return _activeUserObject;
    }
    
    public double GetCurrentThreshold()
    {
        return _currentThreshold;
    }
    
    public async Task DeleteUser(GameObject clickedGameObject)
    {
        if(_recording) return;
        await uIManager.DeleteUserGameObject(clickedGameObject);
        uIManager.HighlightFirstUser();
    }

    public void ThresholdClicked(int i)
    {
        if(_recording) return;
        _currentThreshold = uIManager.HighlightThreshold(i);
    }
}