using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class ShowInfoPong : MonoBehaviour {
    public PongDQLRunner DQLrunner;
    public PongEnvironment environment;
    public Text stepsText;
    public Text episodeText;
    public Text leftWinText;
    public Text rightWinText;
    public Text leftWinPerText;
    public Text leftHitPercText;
	// Use this for initialization
	void Start () {
		
	}

    // Update is called once per frame
    void Update() {
        if (DQLrunner)
        {
            stepsText.text = "Total Steps: " + DQLrunner.Steps.ToString();
            episodeText.text = "Total Episodes: " + DQLrunner.currentEpisode.ToString();
            leftWinText.text = DQLrunner.leftWin.ToString();
            rightWinText.text = DQLrunner.rightWin.ToString();
            leftWinPerText.text = "Win Rate: " + (DQLrunner.winningRate100Left.Average).ToString();
            leftHitPercText.text = "Ave Reward: " + DQLrunner.reward100EpiLeft.Average.ToString();
        }
    }
}
