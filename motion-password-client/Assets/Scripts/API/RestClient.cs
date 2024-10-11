using System;
using System.IO;
using System.Net.Http;
using System.Threading.Tasks;
using UnityEngine;

namespace API
{
    public class RestClient
    {
        private readonly HttpClient _httpClient;
        private readonly string _baseUrl;

        public RestClient(string baseUrl)
        {
            _baseUrl = baseUrl;
            _httpClient = new HttpClient();
        }

        public async Task<HttpResponseMessage> GetAsync(string endpoint)
        {
            try
            {
                var response = await _httpClient.GetAsync($"{_baseUrl}/{endpoint}");
                return response;
            }
            catch (Exception ex)
            {
                Debug.Log("Exception: " + ex.Message);
                return null;
            }
        }

        public async Task<HttpResponseMessage> PostAsync(string endpoint, string jsonContent)
        {
            try
            {
                var content = new StringContent(jsonContent, null, "application/json");
                var response = await _httpClient.PostAsync($"{_baseUrl}/{endpoint}", content);
                return response;
            }
            catch (Exception ex)
            {
                Debug.Log("Exception: " + ex.Message);
                return null;
            }
        }

        public async Task<HttpResponseMessage> DeleteAsync(string endpoint)
        {
            try
            {
                var response = await _httpClient.DeleteAsync($"{_baseUrl}/{endpoint}");
                return response;

            }
            catch (Exception ex)
            {
                Debug.Log("Exception: " + ex.Message);
                return null;
            }
        }

        public async Task<HttpResponseMessage> PostFileAsync(string endpoint, string filePath)
        {
            try
            {
                using var formData = new MultipartFormDataContent();
                await using var fileStream = File.OpenRead(filePath);
                var fileContent = new StreamContent(fileStream);
                var fileName = Path.GetFileName(filePath);
                formData.Add(fileContent, name:"file", fileName:fileName);

                var response = await _httpClient.PostAsync($"{_baseUrl}/{endpoint}", formData);

                return response;
            }
            catch (Exception ex)
            {
                Debug.Log("Exception: " + ex.Message);
                return null;
            }
        }
    
    }
}

