import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

class SpotifyService:
    """Service to interact with Spotify Web API using Spotipy"""
    
    def __init__(self):
        """Initialize the Spotify service with credentials"""
        self.client_id = os.environ.get('SPOTIFY_CLIENT_ID')
        self.client_secret = os.environ.get('SPOTIFY_CLIENT_SECRET')
        
        if not self.client_id or not self.client_secret:
            print("WARNING: Spotify credentials not set. Please set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET environment variables.")
            self.sp = None
        else:
            # Initialize Spotipy client
            auth_manager = SpotifyClientCredentials(
                client_id=self.client_id, 
                client_secret=self.client_secret
            )
            self.sp = spotipy.Spotify(auth_manager=auth_manager)
            print("Spotify service initialized with Spotipy")
        
        # Emotion to music mapping - using standard Spotify genres
        self.emotion_to_music = {
            'happy': {'seed_genres': 'pop', 'target_valence': 0.8, 'target_energy': 0.8},
            'sad': {'seed_genres': 'piano', 'target_valence': 0.2, 'target_energy': 0.3},
            'angry': {'seed_genres': 'rock', 'target_valence': 0.4, 'target_energy': 0.9},
            'surprised': {'seed_genres': 'electronic', 'target_valence': 0.7, 'target_energy': 0.8},
            'fearful': {'seed_genres': 'classical', 'target_valence': 0.3, 'target_energy': 0.4},
            'disgusted': {'seed_genres': 'punk', 'target_valence': 0.3, 'target_energy': 0.8},
            'neutral': {'seed_genres': 'indie', 'target_valence': 0.5, 'target_energy': 0.5},
            'tired': {'seed_genres': 'classical', 'target_valence': 0.4, 'target_energy': 0.2}
        }
    
    def get_recommendations_for_emotion(self, emotion, limit=3):
        """
        Get music recommendations based on emotion
        
        Args:
            emotion (str): The detected emotion
            limit (int): Number of recommendations to return
            
        Returns:
            list: List of track recommendations
        """
        if not self.sp:
            return {"error": "Spotify client not initialized"}
            
        if emotion not in self.emotion_to_music:
            print(f"Unknown emotion: {emotion}. Defaulting to neutral.")
            emotion = 'neutral'
        
        try:
            # Get music parameters for the emotion
            music_params = self.emotion_to_music[emotion]
            
            # Try with seed genres
            print(f"Getting recommendations for {emotion}")
            try:
                # Get recommendations using spotipy
                results = self.sp.recommendations(
                    seed_genres=[music_params['seed_genres']], 
                    target_valence=music_params['target_valence'],
                    target_energy=music_params['target_energy'],
                    limit=limit
                )
                
                # If we get results, format them
                if results and 'tracks' in results:
                    return self._format_track_results(results['tracks'])
            except Exception as e:
                print(f"Error with genre seed: {str(e)}")
            
            # If that fails, try with a popular artist
            print("Trying with artist seed")
            try:
                # Search for a popular artist
                artist_results = self.sp.search(q='artist:Drake', type='artist', limit=1)
                if artist_results and 'artists' in artist_results and artist_results['artists']['items']:
                    artist_id = artist_results['artists']['items'][0]['id']
                    
                    # Get recommendations using the artist
                    results = self.sp.recommendations(
                        seed_artists=[artist_id],
                        limit=limit
                    )
                    
                    # If we get results, format them
                    if results and 'tracks' in results:
                        return self._format_track_results(results['tracks'])
            except Exception as e:
                print(f"Error with artist seed: {str(e)}")
            
            # Last resort: search for tracks based on emotion
            print("Using track search as fallback")
            search_term = f"{emotion} music"
            search_results = self.sp.search(q=search_term, type='track', limit=limit)
            
            if search_results and 'tracks' in search_results:
                return self._format_track_results(search_results['tracks']['items'])
                
            # If all else fails, return empty tracks array
            return {'tracks': []}
            
        except Exception as e:
            print(f"Error getting recommendations: {str(e)}")
            return {"error": str(e), "tracks": []}
    
    def _format_track_results(self, tracks):
        """
        Format track results
        
        Args:
            tracks (list): List of track objects from Spotify API
            
        Returns:
            dict: Formatted tracks
        """
        formatted_tracks = []
        
        for track in tracks:
            artist_names = ", ".join([artist['name'] for artist in track['artists']])
            
            track_info = {
                'id': track['id'],
                'title': track['name'],
                'artist': artist_names,
                'album': track['album']['name'],
                'image_url': track['album']['images'][0]['url'] if track['album']['images'] else None,
                'preview_url': track['preview_url'],
                'external_url': track['external_urls']['spotify']
            }
            formatted_tracks.append(track_info)
            
        return {'tracks': formatted_tracks}