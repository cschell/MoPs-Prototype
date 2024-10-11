namespace API
{
    [System.Serializable]
    public class UserList
    {
        public User[] users;
    }

    [System.Serializable]
    public class User
    {
        public int id;
        public string username;
        public string created_at;
    }
    
}