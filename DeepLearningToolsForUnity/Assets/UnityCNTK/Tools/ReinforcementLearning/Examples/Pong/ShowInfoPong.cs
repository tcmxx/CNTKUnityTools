using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class ShowInfoPong : MonoBehaviour {
    public PongDQLRunner DQLrunner;
    public PongPPORunner PPOrunner;
    public PongEnvironment environment;
    public Text stepsText;
    public Text episodeText;
    public Text leftWinText;
    public Text rightWinText;
    public Text leftWinPerText;
    public Text leftHitPercText;
    public Text infoText;
	// Use this for initialization
	void Start () {
		
	}

    // Update is called once per frame
    void Update() {
        if (DQLrunner && DQLrunner.isActiveAndEnabled)
        {
            stepsText.text = "Total Steps: " + DQLrunner.Steps.ToString();
            episodeText.text = "Total Episodes: " + DQLrunner.currentEpisode.ToString();
            leftWinText.text = DQLrunner.leftWin.ToString();
            rightWinText.text = DQLrunner.rightWin.ToString();
            leftWinPerText.text = "Win Rate: " + (DQLrunner.winningRate50Left.Average).ToString();
            leftHitPercText.text = "Ave Reward: " + DQLrunner.reward50EpiLeft.Average.ToString();
            infoText.gameObject.SetActive(false);
        }
        else if (PPOrunner && PPOrunner.isActiveAndEnabled)
        {
            stepsText.text = "Total Steps: " + PPOrunner.Steps.ToString();
            episodeText.text = "Total Episodes: " + PPOrunner.currentEpisode.ToString();
            leftWinText.text = PPOrunner.leftWin.ToString();
            rightWinText.text = PPOrunner.rightWin.ToString();
            leftWinPerText.text = "Win Rate: " + (PPOrunner.winningRate50Left.Average).ToString();
            leftHitPercText.text = "Ave Reward: " + PPOrunner.episodePointAve.Average.ToString();
            infoText.gameObject.SetActive(true);
            infoText.text = "PPO not working for this yet";
        }
    }
}
