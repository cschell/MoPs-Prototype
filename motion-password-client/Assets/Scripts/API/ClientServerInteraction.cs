using System.Collections;
using System.Globalization;
using System.IO;
using System.IO.Compression;
using System.Threading.Tasks;
using UnityEngine;

namespace API
{
    public class ClientServerInteraction : MonoBehaviour
    {
        private const string BaseUrl = "http://localhost:8000";
        private RestClient _restClient;

        // Called when the script instance is being loaded
        private void Awake()
        {
            Debug.Log("Awake");

            _restClient = new RestClient(BaseUrl);
        }

        // Start is called before the first frame update
        private void Start()
        {
            StartCoroutine(SayHello());

        }
    
        private IEnumerator SayHello()
        {
        
            //PostUser("yusef");

            //GetAllUsers();

            //GetUser(2);

            //DeleteUser(2);

            //GetAllUsers();

            //PostUser("yusef2");

            //DeleteUsername("yusef2");

            //GetAllUsers();
    


            //PostEnrollment(1, 1);
        
            //GetAllEmbeddings(0, 100);

            //GetEmbedding(3);

            //GetAllUserEmbeddings(1);

            //DeleteAllUserEmbeddings(4);

            //DeleteEmbedding(2);

            //GetAuthentication(1, 1);
        
            //GetIdentification(1);
        
            //Getestet

            //PostCreateModel(1, "scripted_model.pt", 1, 1, 1, 1, "cosine_similarity");
        
            //GetModel(1);
        
            //GetAllModels(0, 100);
        
            //DeleteModel(1);
        
        

            yield return new WaitForSeconds(5);
            // ReSharper disable once IteratorNeverReturns
        }

        public async void GetUser(int userID)
        {
            var response = await _restClient.GetAsync("v1/user/" + userID);

            if (response.IsSuccessStatusCode)
            {
                var responseBody = await response.Content.ReadAsStringAsync();
                Debug.Log("Response: " + responseBody);
            }
            else
            {
                Debug.Log("Error: " + response.StatusCode);
                var errorResponse = await response.Content.ReadAsStringAsync();
                Debug.Log("Error Response: " + errorResponse);
            }
        }

        public async Task<string> GetAllUsers()
        {
            var response = await _restClient.GetAsync("v1/read_users");

            if (response.IsSuccessStatusCode)
            {
                var responseBody = await response.Content.ReadAsStringAsync();
                Debug.Log("Response: " + responseBody);
                return responseBody;
            }
            else
            {
                Debug.Log("Error: " + response.StatusCode);
                var errorResponse = await response.Content.ReadAsStringAsync();
                Debug.Log("Error Response: " + errorResponse);
                return errorResponse;
            }
        }

        public async Task<string> PostUser(string username)
        {
            var jsonContent = "{\"username\":\"" + username + "\"}";
            var response = await _restClient.PostAsync("v1/user", jsonContent);

            if (response.IsSuccessStatusCode)
            {
                var responseBody = await response.Content.ReadAsStringAsync();
                Debug.Log("Response: " + responseBody);
                return responseBody;
            }
            else
            {
                Debug.Log("Error: " + response.StatusCode);
                var errorResponse = await response.Content.ReadAsStringAsync();
                Debug.Log("Error Response: " + errorResponse);
                return errorResponse;
            }
        }

        public async void DeleteUser(int userID)
        {
            var response = await _restClient.DeleteAsync("v1/user/" + userID);

            if (response.IsSuccessStatusCode)
            {
                var responseBody = await response.Content.ReadAsStringAsync();
                Debug.Log("Response: " + responseBody);
            }
            else
            {
                Debug.Log("Error: " + response.StatusCode);
                var errorResponse = await response.Content.ReadAsStringAsync();
                Debug.Log("Error Response: " + errorResponse);
            }
        }

        public async Task<string> DeleteUsername(string username)
        {
            var response = await _restClient.DeleteAsync("v1/user/delete_by_username/" + username);

            if (response.IsSuccessStatusCode)
            {
                var responseBody = await response.Content.ReadAsStringAsync();
                Debug.Log("Response: " + responseBody);
                return "Successful";
            }
            else
            {
                Debug.Log("Error: " + response.StatusCode);
                var errorResponse = await response.Content.ReadAsStringAsync();
                Debug.Log("Error Response: " + errorResponse);
                return "Not Successful";
            }
        }
    
        private static string CompressCsvFile(string inputCsvFilename)
        {
            var outputGzipFilename = Path.ChangeExtension(inputCsvFilename, ".csv.gz");

            using var inputFileStream = new FileStream(inputCsvFilename, FileMode.Open, FileAccess.Read);
            using var outputFileStream = new FileStream(outputGzipFilename, FileMode.Create, FileAccess.Write); 
            using var gzipStream = new GZipStream(outputFileStream, CompressionMode.Compress);
            inputFileStream.CopyToAsync(gzipStream);

            return outputGzipFilename;
        }
    
        public async void PostEnrollment(int userID, int modelID, string filePath)
        {
            var compressedFilePath = CompressCsvFile(filePath);
            var response = await _restClient.PostFileAsync("v1/enrollment/" + userID + "?model_id=" + modelID, compressedFilePath);

            if (response.IsSuccessStatusCode)
            {
                var responseBody = await response.Content.ReadAsStringAsync();
                Debug.Log("Response: " + responseBody);
            }
            else
            {
                Debug.Log("Error: " + response.StatusCode);
                var errorResponse = await response.Content.ReadAsStringAsync();
                Debug.Log("Error Response: " + errorResponse);
            }
        }
    
        public async Task<string> PostEnrollmentV2(int userID, int modelID, string filePath)
        {
            //var compressedFilePath = CompressCsvFile(filePath);
            var response = await _restClient.PostFileAsync("v2/enrollment/" + userID + "?model_id=" + modelID, filePath);

            if (response.IsSuccessStatusCode)
            {
                var responseBody = await response.Content.ReadAsStringAsync();
                return "Successful";
                
            }
            else
            {
                Debug.Log("Error: " + response.StatusCode);
                var errorResponse = await response.Content.ReadAsStringAsync();
                return "Not Successful";
            }

        }
    
        public async void GetAuthentication(int userID, int modelID, string filePath)
        {
            var compressedFilePath = CompressCsvFile(filePath);

            var response = await _restClient.PostFileAsync("v1/authentication/" + userID + "?model_id=" + modelID, compressedFilePath);

            if (response.IsSuccessStatusCode)
            {
                var responseBody = await response.Content.ReadAsStringAsync();
                Debug.Log("Response: " + responseBody);
            }
            else
            {
                Debug.Log("Error: " + response.StatusCode);
                var errorResponse = await response.Content.ReadAsStringAsync();
                Debug.Log("Error Response: " + errorResponse);
            }

        }
    
        public async Task<string> GetAuthenticationV2(int userID, int modelID, double threshold, string filePath)
        {
            //var compressedFilePath = CompressCsvFile(filePath);
            // Format the threshold as a dot-separated string
            var formattedThreshold = threshold.ToString(CultureInfo.InvariantCulture); // Use a culture-invariant format

            var response = await _restClient.PostFileAsync("v2/authentication/" + userID + "?model_id=" + modelID + "&threshold=" + formattedThreshold, filePath);

            if (response.IsSuccessStatusCode)
            {
                var responseBody = await response.Content.ReadAsStringAsync();
                return ("Response: " + responseBody);
            }
            else
            {
                Debug.Log("Error: " + response.StatusCode);
                var errorResponse = await response.Content.ReadAsStringAsync();
                return ("Error Response: " + errorResponse);
            }
        }
    
        public async void GetIdentification(int modelID, string filePath)
        {
            //var compressedFilePath = CompressCsvFile(filePath);
        
            var response = await _restClient.PostFileAsync("v1/identification/" + "?model_id=" + modelID, filePath);

            if (response.IsSuccessStatusCode)
            {
                var responseBody = await response.Content.ReadAsStringAsync();
                Debug.Log("Response: " + responseBody);
            }
            else
            {
                Debug.Log("Error: " + response.StatusCode);
                var errorResponse = await response.Content.ReadAsStringAsync();
                Debug.Log("Error Response: " + errorResponse);
            }

        }
    
        public async Task<string> GetIdentificationV2(int modelID, string filePath)
        {
            //var compressedFilePath = CompressCsvFile(filePath);
        
            var response = await _restClient.PostFileAsync("v2/identification/" + "?model_id=" + modelID, filePath);

            if (response.IsSuccessStatusCode)
            {
                var responseBody = await response.Content.ReadAsStringAsync();
                return (responseBody);
            }
            else
            {
                Debug.Log("Error: " + response.StatusCode);
                var errorResponse = await response.Content.ReadAsStringAsync();
                Debug.Log("Error Response: " + errorResponse);
                return ("Error Response: " + errorResponse);
            }

        }
    
        public async void GetAllEmbeddings(int skip, int limit)
        {
            var response = await _restClient.GetAsync("v1/read_embeddings/?skip=" + skip + "&limit=" + limit);

            if (response.IsSuccessStatusCode)
            {
                var responseBody = await response.Content.ReadAsStringAsync();
                Debug.Log("Response: " + responseBody);
            }
            else
            {
                Debug.Log("Error: " + response.StatusCode);
                var errorResponse = await response.Content.ReadAsStringAsync();
                Debug.Log("Error Response: " + errorResponse);
            }
        }

        public async void GetEmbedding(int embeddingID)
        {
            var response = await _restClient.GetAsync("v1/embedding/" + embeddingID);

            if (response.IsSuccessStatusCode)
            {
                var responseBody = await response.Content.ReadAsStringAsync();
                Debug.Log("Response: " + responseBody);
            }
            else
            {
                Debug.Log("Error: " + response.StatusCode);
                var errorResponse = await response.Content.ReadAsStringAsync();
                Debug.Log("Error Response: " + errorResponse);
            }
        }

        public async Task<string> GetAllUserEmbeddings(int userID)
        {
            var response = await _restClient.GetAsync("v1/user/" + userID + "/embedding");

            if (response.IsSuccessStatusCode)
            {
                var responseBody = await response.Content.ReadAsStringAsync();
                Debug.Log("Response: " + responseBody);
                return responseBody;
            }
            else
            {
                Debug.Log("Error: " + response.StatusCode);
                var errorResponse = await response.Content.ReadAsStringAsync();
                Debug.Log("Error Response: " + errorResponse);
                return errorResponse;
            }
        }

        public async void DeleteEmbedding(int embeddingID)
        {
            var response = await _restClient.DeleteAsync("v1/embedding/" + embeddingID);

            if (response.IsSuccessStatusCode)
            {
                var responseBody = await response.Content.ReadAsStringAsync();
                Debug.Log("Response: " + responseBody);
            }
            else
            {
                Debug.Log("Error: " + response.StatusCode);
                var errorResponse = await response.Content.ReadAsStringAsync();
                Debug.Log("Error Response: " + errorResponse);
            }
        }

        public async void DeleteAllUserEmbeddings(int userID)
        {
            var response = await _restClient.DeleteAsync("v1/all_embeddings/user/" + userID);

            if (response.IsSuccessStatusCode)
            {
                var responseBody = await response.Content.ReadAsStringAsync();
                Debug.Log("Response: " + responseBody);
            }
            else
            {
                Debug.Log("Error: " + response.StatusCode);
                var errorResponse = await response.Content.ReadAsStringAsync();
                Debug.Log("Error Response: " + errorResponse);
            }
        }
    
    
        public async void PostCreateModel(int userID, string modelName, int version, int windowSize, int overlap, int fps, string distanceAlgorithm, string filePath)
        {
            var compressedFilePath = CompressCsvFile(filePath);

            var endpoint = $"v1/model?id={userID}&name={modelName}&version={version}&window_size={windowSize}&overlap={overlap}&fps={fps}&distance_algorithm={distanceAlgorithm}";
        
            var response = await _restClient.PostFileAsync(endpoint, compressedFilePath);

            if (response.IsSuccessStatusCode)
            {
                var responseBody = await response.Content.ReadAsStringAsync();
                Debug.Log("Response: " + responseBody);
            }
            else
            {
                Debug.Log("Error: " + response.StatusCode);
                var errorResponse = await response.Content.ReadAsStringAsync();
                Debug.Log("Error Response: " + errorResponse);
            }

        }
    
        public async void GetAllModels(int skip, int limit)
        {
            var response = await _restClient.GetAsync("v1/model?skip=" + skip + "&limit=" + limit);

            if (response.IsSuccessStatusCode)
            {
                var responseBody = await response.Content.ReadAsStringAsync();
                Debug.Log("Response: " + responseBody);
            }
            else
            {
                Debug.Log("Error: " + response.StatusCode);
                var errorResponse = await response.Content.ReadAsStringAsync();
                Debug.Log("Error Response: " + errorResponse);
            }
        }
    
    
        public async void GetModel(int modelID)
        {
            var response = await _restClient.GetAsync("v1/model/" + modelID );

            if (response.IsSuccessStatusCode)
            {
                var responseBody = await response.Content.ReadAsStringAsync();
                Debug.Log("Response: " + responseBody);
            }
            else
            {
                Debug.Log("Error: " + response.StatusCode);
                var errorResponse = await response.Content.ReadAsStringAsync();
                Debug.Log("Error Response: " + errorResponse);
            }
        }
    
        public async void DeleteModel(int modelId)
        {
            var response = await _restClient.DeleteAsync("v1/model/" + modelId);

            if (response.IsSuccessStatusCode)
            {
                var responseBody = await response.Content.ReadAsStringAsync();
                Debug.Log("Response: " + responseBody);
            }
            else
            {
                Debug.Log("Error: " + response.StatusCode);
                var errorResponse = await response.Content.ReadAsStringAsync();
                Debug.Log("Error Response: " + errorResponse);
            }
        }
    
    }
}


