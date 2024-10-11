namespace API
{
    [System.Serializable]
    public class EmbeddingList
    {
        public Embedding[] embeddings;
    }
    
    [System.Serializable]
    public class Embedding
    {
        public int quality;
        public int id;
        public int owner_id;
        public string created_at;
        public int model_id;
    }
    
}