using System;
using System.Threading.Tasks;
using API;
using TMPro;
using UnityEditor;
using UnityEngine;
using UnityEngine.Serialization;
using UnityEngine.UI;
using Button = BNG.Button;

public class UIManager : MonoBehaviour
{
    public Button buttonValid;
    public Button buttonInvalid;

    public GameObject newUsersButton;
    public GameObject thresholdValuesButtonsDescription;
    
    public GameObject verticalElement;

    public GameObject imageResponse;
    public GameObject distanceText;

    public GameObject threshold;
    
    public ClientServerInteraction clientServerInteraction;
    public StateManager stateManager;

    public TMP_Text stopWatch;

    private bool _firstInitialization;
    private int _currentUser;

    public async void Start()
    {
        _firstInitialization = true;
        await UpdateUsers();
    }

    private async Task UpdateUsers()
    {
        var response = await clientServerInteraction.GetAllUsers();
        // Wrap ssd within an object containing the "users" array
        var wrappedJson = "{\"users\":" + response + "}";
        var userList = JsonUtility.FromJson<UserList>(wrappedJson);
        
        foreach (var user in userList.users)
        {
            var horizontalElement = verticalElement.transform.GetChild(0).gameObject;
            var copy = Helper.CopyGameObject(horizontalElement,true, user.created_at);
            Helper.SetParent(copy, verticalElement);
            
            var button = copy.transform.GetChild(0).gameObject;
            
            var responseEmbeddings = await clientServerInteraction.GetAllUserEmbeddings(user.id);
            var wrappedJsonEmbedding = "{\"embeddings\":" + responseEmbeddings + "}";
            var embeddingList = JsonUtility.FromJson<EmbeddingList>(wrappedJsonEmbedding);

            var textID = button.transform.GetChild(0).GetComponent<TMP_Text>(); 
            textID.text = user.id.ToString(); 
            
            var textUsername = button.transform.GetChild(1).GetComponent<TMP_Text>();
            textUsername.text = user.username; 
            
            var embeddings = button.transform.GetChild(2).GetComponent<TMP_Text>();
            embeddings.text = embeddingList.embeddings.Length.ToString();

            if (!_firstInitialization) continue;
            stateManager.SetActiveUser(copy, user.id.ToString(), user.username);
            _firstInitialization = false;

        }
        Helper.SetGameObjectActive(verticalElement.transform.GetChild(0).gameObject, false);
    }

    public void ChangeButtonAccessibility(bool buttonAccessibility)
    {
        if (buttonAccessibility)
        {
            Helper.SetGameObjectActive(buttonValid.gameObject, true);
            Helper.SetGameObjectActive(buttonInvalid.gameObject, false);
        }
        else
        {
            Helper.SetGameObjectActive(buttonValid.gameObject, false);
            Helper.SetGameObjectActive(buttonInvalid.gameObject, true);
        }
        
    }

    public void ChangeTime(int minutes, int seconds, int milliseconds, Color color)
    {
        // Update the stopwatch text with formatted time
        stopWatch.text = $"{minutes:00}:{seconds:00}.{milliseconds:00}";
        stopWatch.color = color;
    }

    public void UpdateUI(StateManager.State currentState)
    {
        switch (currentState)
        {
            case StateManager.State.Enroll:
                Helper.SetGameObjectActive(newUsersButton, true);
                Helper.SetGameObjectActive(thresholdValuesButtonsDescription, false);
                break;
            case StateManager.State.Verify:
                Helper.SetGameObjectActive(newUsersButton, false);
                Helper.SetGameObjectActive(thresholdValuesButtonsDescription, true);
                stateManager.ThresholdClicked(1);
                break;
            default:
                Debug.LogError("Illegal State inside UpdateUI()");
                break;
        }
    }

    public double HighlightThreshold(int thresholdID)
    {
        var first = threshold.transform.GetChild(1).gameObject;
        var second = threshold.transform.GetChild(2).gameObject;
        var third = threshold.transform.GetChild(3).gameObject;
        var fourth = threshold.transform.GetChild(4).gameObject;

        var highlightedColor = new Color32(195,156,102,255);
        
        var normalColor = new Color32(137,96,38,255);
        
        double  selectedValue = 0;
        
        switch (thresholdID)
        {
            case 1:
                selectedValue = 0.903;
                first.GetComponent<Image>().color = highlightedColor;
                second.GetComponent<Image>().color = normalColor;
                third.GetComponent<Image>().color = normalColor;
                fourth.GetComponent<Image>().color = normalColor;
                break;
            case 2:
                selectedValue = 0.850;
                first.GetComponent<Image>().color = normalColor;
                second.GetComponent<Image>().color = highlightedColor;
                third.GetComponent<Image>().color = normalColor;
                fourth.GetComponent<Image>().color = normalColor;
                break;
            case 3:
                selectedValue = 0.746;
                first.GetComponent<Image>().color = normalColor;
                second.GetComponent<Image>().color = normalColor;
                third.GetComponent<Image>().color = highlightedColor;
                fourth.GetComponent<Image>().color = normalColor;
                break;
            case 4:
                selectedValue = 0.660;
                first.GetComponent<Image>().color = normalColor;
                second.GetComponent<Image>().color = normalColor;
                third.GetComponent<Image>().color = normalColor;
                fourth.GetComponent<Image>().color = highlightedColor;
                break;
            default:
                selectedValue = 0;
                break;
        }

        return selectedValue;
    }

    public void HighlightActiveUser(GameObject clickedGameObject)
    {
        clickedGameObject.GetComponent<RawImage>().enabled = true;
    }

    public void WashoutHighlights()
    {
        if (verticalElement != null)
        {
            var layoutGroups = verticalElement.GetComponentsInChildren<VerticalLayoutGroup>(true); // Include inactive elements

            if (layoutGroups.Length > 0)
            {
                foreach (var layoutGroup in layoutGroups)
                {
                    var childCount = layoutGroup.transform.childCount;
                    for (var i = 0; i < childCount; i++)
                    {
                        layoutGroup.transform.GetChild(i).GetComponent<RawImage>().enabled = false;
                    }
                }
            }
            else
            {
                Debug.Log("No VerticalLayoutGroup components found within the parent.");
            }
        }
        else
        {
            Debug.LogError("verticalElement reference is null!");
        }
    }

    public async Task CreateNewUser(string username)
    {
        var response = await clientServerInteraction.PostUser(username);
        
        var id = Helper.GetID(response);
        
        var horizontalElement = verticalElement.transform.GetChild(0).gameObject;
        var copy = Helper.CopyGameObject(horizontalElement,true, id);
        Helper.SetParent(copy, verticalElement);
            
        var button = copy.transform.GetChild(0).gameObject;
            
        var responseEmbeddings = await clientServerInteraction.GetAllUserEmbeddings(int.Parse(id));
        var wrappedJsonEmbedding = "{\"embeddings\":" + responseEmbeddings + "}";
        var embeddingList = JsonUtility.FromJson<EmbeddingList>(wrappedJsonEmbedding);

        var textID = button.transform.GetChild(0).GetComponent<TMP_Text>(); 
        textID.text = id; 
            
        var textUsername = button.transform.GetChild(1).GetComponent<TMP_Text>();
        textUsername.text = username; 
        
        var embeddings = button.transform.GetChild(2).GetComponent<TMP_Text>();
        embeddings.text = embeddingList.embeddings.Length.ToString();
        
        Helper.SetGameObjectActive(copy, true);
        stateManager.SetActiveUser(copy, id, username);
    }
    
    public void HighlightFirstUser()
    {
        
        var horizontalElement = verticalElement.transform.GetChild(1).gameObject;
        
        var button = horizontalElement.transform.GetChild(0).gameObject;
        var textID = button.transform.GetChild(0).GetComponent<TMP_Text>(); 
        var textUsername = button.transform.GetChild(1).GetComponent<TMP_Text>();

        stateManager.SetActiveUser(horizontalElement, textID.text, textUsername.text);

    }

    public async Task DeleteUserGameObject(GameObject clickedGameObject)
    {
        var button = clickedGameObject.transform.GetChild(0).gameObject;
        var textUsername = button.transform.GetChild(1).GetComponent<TMP_Text>();

        var response = await clientServerInteraction.DeleteUsername(textUsername.text);
        
        if (response.Equals("Successful"))
        {
            Helper.SetGameObjectActive(imageResponse.transform.GetChild(0).gameObject, true);
            Helper.SetGameObjectActive(imageResponse.transform.GetChild(1).gameObject, false);
        }
        else
        {
            Helper.SetGameObjectActive(imageResponse.transform.GetChild(0).gameObject, false);
            Helper.SetGameObjectActive(imageResponse.transform.GetChild(1).gameObject, true);
        }
        
        Destroy(clickedGameObject);
    }

    public async Task SendData()
    {
        var data = Helper.GetLatestFile();
        var responseText = "";
        var verify = false;
        switch (stateManager.GetCurrentState())
        {
            case StateManager.State.Enroll:
                responseText = await clientServerInteraction.PostEnrollmentV2(stateManager.GetActiveUser().id, modelID:1, data);
                var user = stateManager.GetActiveUserObject();
                
                var button = user.transform.GetChild(0).gameObject;
            
                var responseEmbeddings = await clientServerInteraction.GetAllUserEmbeddings(stateManager.GetActiveUser().id);
                var wrappedJsonEmbedding = "{\"embeddings\":" + responseEmbeddings + "}";
                var embeddingList = JsonUtility.FromJson<EmbeddingList>(wrappedJsonEmbedding);
                
                var embeddings = button.transform.GetChild(2).GetComponent<TMP_Text>();
                embeddings.text = embeddingList.embeddings.Length.ToString();
                
                break;
            case StateManager.State.Verify:
                responseText = await clientServerInteraction.GetAuthenticationV2(stateManager.GetActiveUser().id, 1, stateManager.GetCurrentThreshold(), data);
                distanceText.GetComponent<TMP_Text>().text = responseText;
                verify = true;
                break;
            default:
                Debug.LogError("Illegal State inside UpdateUI()");
                break;
        }
        
        if (responseText.Contains("Successful"))
        {
            Helper.SetGameObjectActive(imageResponse.transform.GetChild(0).gameObject, true);
            Helper.SetGameObjectActive(imageResponse.transform.GetChild(1).gameObject, false);
        }
        else
        {
            Helper.SetGameObjectActive(imageResponse.transform.GetChild(0).gameObject, false);
            Helper.SetGameObjectActive(imageResponse.transform.GetChild(1).gameObject, true);
        }

        Helper.SetGameObjectActive(distanceText, verify);
    }
}
