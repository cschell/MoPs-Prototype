using System;
using System.Collections.Generic;

public static class RandomUsernameGenerator
{
    // List of potential username components
    private static readonly string[] Adjectives = { "Amazing", "Brilliant", "Curious", "Dazzling", "Energetic", "Fearless" }; 
    private static readonly string[] Nouns = { "Badger", "Owl", "Dragonfly", "Fox", "Whale", "Eagle", "Lion", "Turtle", }; 
    private static readonly string[] Numbers = { "1", "2", "3", "4", "5", "6", "7", "8", "9", "0" };

    // Create a list of random usernames
    private static readonly List<string> RandomUsernames = new List<string>();

    static RandomUsernameGenerator()
    {
        // Generate a list of 50 unique usernames (adjust as needed)
        for (var i = 0; i < 50; i++)
        {
            var username = GenerateUniqueUsername();
            RandomUsernames.Add(username);
        }
    }

    // Function to create a unique username with format "adjective-noun-number"
    private static string GenerateUniqueUsername()
    {
        var random = new Random();
        var adjective = Adjectives[random.Next(Adjectives.Length)];
        var noun = Nouns[random.Next(Nouns.Length)];
        var number = Numbers[random.Next(Numbers.Length)];
        var username = $"{adjective.ToLowerInvariant()}_{noun.ToLowerInvariant()}_{number}";

        // Check for uniqueness (optional, for larger lists)
        while (RandomUsernames.Contains(username))
        {
            username = GenerateUniqueUsername();
        }

        return username;
    }

    public static string GetNewRandomUsername()
    {
        var random = new Random();
        var username = RandomUsernames[random.Next(RandomUsernames.Count)];
        RandomUsernames.Remove(username);
        return username;
    }
}
