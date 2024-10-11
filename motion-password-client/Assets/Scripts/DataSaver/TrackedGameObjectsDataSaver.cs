using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;

namespace UserMovementRecording
{
    [RequireComponent(typeof(TrackGameObjects))]
    public class TrackedGameObjectsDataSaver : MonoBehaviour
    {
        public string pathToCsv = "MotionData"; // Path to the CSV File
        
        public int chunkSizeInSeconds = 10;
        
        private static TrackedGameObjectsDataSaver _gameObjectsDataSaver;
        private TrackGameObjects _trackGameObjects;
        
        private string _dateTime;
        private bool _isRecording;
        
        private readonly Queue<string> _dataRows = new();
        
        private Coroutine _saveDataRoutine;
        private Coroutine _addDataRoutine;
        private void Awake()
        {
            if (_gameObjectsDataSaver != null)
            {
                Destroy(gameObject);
                return;
            }

            UpdateTimeCsvPath();
            
            _gameObjectsDataSaver = this;
            _isRecording = false;
            DontDestroyOnLoad(this);
            
            _trackGameObjects = GetComponent<TrackGameObjects>();
        }

        
        public void UpdateTimeCsvPath()
        {
            _dateTime = DateTime.Now.ToString("yyyy-M-d_HH-mm-ss");
            pathToCsv = Path.Join(pathToCsv, $"movement_{_dateTime}.csv");
            
            if (!File.Exists(pathToCsv))
            {
                AddHeaderLine();
            }
        }
        
        public void StartRecording()
        {
            if (_isRecording) return;
            _addDataRoutine = StartCoroutine(AddDataCoroutine());
            _saveDataRoutine = StartCoroutine(SaveDataCoroutine());
            _isRecording = true;
        }

        public void StopRecording()
        {
            if (!_isRecording) return;
            StopCoroutine(_addDataRoutine);
            StopCoroutine(_saveDataRoutine);
            SaveAllData();
            _isRecording = false;
        }

        public bool IsRecording()
        {
            return _isRecording;
        }
        
        private void SaveAllData()
        {
            if (_dataRows.Count <= 0) return;

            var currentRowCount = _dataRows.Count;
            using var writer = new StreamWriter(pathToCsv, true);
            for (var i = 0; i < currentRowCount; i++)
            {
                writer.WriteLine(_dataRows.Dequeue());
            }
        }
        
        public void UpdateChunkSize(int newChunkSizeInSeconds)
        {
            chunkSizeInSeconds = newChunkSizeInSeconds;
        }
        
        
        private void AddHeaderLine()
        {
            using var writer = new StreamWriter(pathToCsv, true);
            //// Write the header line
            {
                writer.WriteLine(
                    "Timestamp," +
                    
                    "DevicePositionX,DevicePositionY,DevicePositionZ," +
                    "DeviceRotationX,DeviceRotationY,DeviceRotationZ,DeviceRotationW," +
                    
                    "RightControllerPositionX,RightControllerPositionY,RightControllerPositionZ," +
                    "RightControllerRotationX,RightControllerRotationY,RightControllerRotationZ,RightControllerRotationW," +
                    
                    "LeftControllerPositionX,LeftControllerPositionY,LeftControllerPositionZ," +
                    "LeftControllerRotationX,LeftControllerRotationY,LeftControllerRotationZ,LeftControllerRotationW"
                    );

            }
        }
        
        private IEnumerator SaveDataCoroutine()
        {
            while (true)
            {
                yield return new WaitForSeconds(chunkSizeInSeconds);

                SaveAllData();
            }
            // ReSharper disable once IteratorNeverReturns
        }
        
        private IEnumerator AddDataCoroutine()
        {
            while (true)
            {
                var (headDataRow, leftDataRow, rightDataRow) = GatherDataIntoLists();

                AddDataToRows(headDataRow, leftDataRow, rightDataRow);
                
            }
            // ReSharper disable once IteratorNeverReturns
        }

        

        
        private void AddDataToRows(DataRow headDataRow, DataRow leftDataRow, DataRow rightDataRow)
        {
            StringBuilder stringBuilder = new StringBuilder();

            stringBuilder.Append(DateTime.Now.ToString("T", CultureInfo.InvariantCulture));
            stringBuilder.Append(",");
            stringBuilder.Append(headDataRow.DevicePosition);
            stringBuilder.Append(",");
            stringBuilder.Append(headDataRow.DeviceRotation);
            stringBuilder.Append(",");
            
            stringBuilder.Append(rightDataRow.DevicePosition);
            stringBuilder.Append(",");
            stringBuilder.Append(rightDataRow.DeviceRotation);
            stringBuilder.Append(",");
                
            stringBuilder.Append(leftDataRow.DevicePosition);
            stringBuilder.Append(",");
            stringBuilder.Append(leftDataRow.DeviceRotation);

            _dataRows.Enqueue(stringBuilder.ToString());
        }

        private (DataRow, DataRow, DataRow)  GatherDataIntoLists()
        {
            DataRow headDataRow = new DataRow();

            headDataRow = GetData(headDataRow, _trackGameObjects.hmd);
            
            DataRow leftDataRow = new DataRow();

            leftDataRow = GetData(leftDataRow, _trackGameObjects.hmd);
            
            DataRow rightDataRow = new DataRow();

            rightDataRow = GetData(rightDataRow, _trackGameObjects.hmd);
        
            return (headDataRow, leftDataRow, rightDataRow);
        }

        private DataRow GetData(DataRow dataRow, GameObject gameObject)
        {
            dataRow.DevicePosition = ReformatClean(gameObject.transform.position.ToString());
            dataRow.DeviceRotation = ReformatClean(gameObject.transform.rotation.ToString());

            return dataRow;
        }

        private string ReformatClean(string dirty)
        {
            StringBuilder clean = new StringBuilder (dirty);

            clean.Replace("(", "");
            clean.Replace(")", "");
            //clean.Replace(",", ";");

            return clean.ToString();
        }

        private class DataRow
        {
            public string DevicePosition { get; set; }
            public string DeviceRotation { get; set; }
        }

    }
}