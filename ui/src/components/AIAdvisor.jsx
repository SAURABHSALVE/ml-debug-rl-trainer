import { useState, useEffect } from 'react';
import './AIAdvisor.css';

export default function AIAdvisor({ taskDifficulty, stepsUsed, stepsRemaining, onActionSelected }) {
  const [recommendations, setRecommendations] = useState(null);
  const [diagnosis, setDiagnosis] = useState(null);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('recommend'); // 'recommend' | 'suggest' | 'learn'

  const API = '/api';

  useEffect(() => {
    if (taskDifficulty) {
      fetchRecommendations();
      fetchStats();
    }
  }, [taskDifficulty, stepsUsed]);

  const fetchRecommendations = async () => {
    try {
      const res = await fetch(`${API}/recommend-actions?limit=5`, { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        setRecommendations(data);
      }
    } catch (e) {
      console.error('Failed to fetch recommendations:', e);
    }
  };

  const fetchSuggestion = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API}/suggest-diagnosis`, { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        setDiagnosis(data);
      }
    } catch (e) {
      console.error('Failed to fetch suggestion:', e);
    } finally {
      setLoading(false);
    }
  };

  const fetchStats = async () => {
    try {
      const res = await fetch(`${API}/agent-stats`);
      if (res.ok) {
        const data = await res.json();
        setStats(data);
      }
    } catch (e) {
      console.error('Failed to fetch stats:', e);
    }
  };

  const handleActionClick = (actionName) => {
    if (onActionSelected) {
      onActionSelected(actionName);
    }
  };

  const actionEmojis = {
    fetch_logs: '📋',
    fetch_config: '⚙️',
    fetch_loss_curve: '📈',
    fetch_diagnostics: '🔍',
    fetch_class_data: '📊',
    diagnose: '🎯',
  };

  return (
    <div className="ai-advisor">
      <div className="advisor-header">
        <h3>🤖 AI Debugging Coach</h3>
        <p className="subtitle">What the RL agent learned from {stats?.total_training_episodes || 0} debugging sessions</p>
      </div>

      <div className="advisor-tabs">
        <button 
          className={`tab-btn ${activeTab === 'recommend' ? 'active' : ''}`}
          onClick={() => setActiveTab('recommend')}
        >
          💡 Smart Moves
        </button>
        <button 
          className={`tab-btn ${activeTab === 'suggest' ? 'active' : ''}`}
          onClick={() => { setActiveTab('suggest'); fetchSuggestion(); }}
        >
          🎯 Stuck? Get Hint
        </button>
        <button 
          className={`tab-btn ${activeTab === 'learn' ? 'active' : ''}`}
          onClick={() => setActiveTab('learn')}
        >
          📚 What Agent Learned
        </button>
      </div>

      {activeTab === 'recommend' && recommendations && (
        <div className="advisor-content">
          <div className="step-meter">
            <div className="meter-label">Steps Used: {stepsUsed} / 15</div>
            <div className="meter-bar">
              <div className="meter-fill" style={{ width: `${(stepsUsed / 15) * 100}%` }}></div>
            </div>
          </div>

          <p className="advisor-text">
            Based on successful debugging patterns, try these actions next:
          </p>
          <div className="recommended-actions">
            {recommendations.recommended_actions.map((action, idx) => (
              <button 
                key={action}
                className={`action-card rank-${idx + 1}`}
                onClick={() => handleActionClick(action)}
                title={`Recommended step ${idx + 1}`}
              >
                <div className="rank">{idx + 1}</div>
                <span className="emoji">{actionEmojis[action] || '→'}</span>
                <span className="action-name">{action.replace('_', ' ')}</span>
                <span className="confidence">
                  {idx === 0 ? '⭐ Highest value' : `${100 - idx * 15}% match`}
                </span>
              </button>
            ))}
          </div>
          <p className="advisor-note">💭 {recommendations.reasoning}</p>
        </div>
      )}

      {activeTab === 'suggest' && diagnosis && (
        <div className="advisor-content">
          <div className="diagnosis-box">
            <h4>🔎 Investigation Direction</h4>
            <p><strong>Expected Issue:</strong> {diagnosis.expected_bug_type.replace(/_/g, ' ')}</p>
            <p className="hint">{diagnosis.investigation_hint}</p>
            
            <h4 style={{ marginTop: '1rem' }}>🔧 Likely Fixes</h4>
            <ul className="fix-list">
              {diagnosis.fix_suggestions?.map(fix => (
                <li key={fix}>{fix.replace(/_/g, ' ')}</li>
              ))}
            </ul>
          </div>
        </div>
      )}

      {activeTab === 'suggest' && !diagnosis && (
        <div className="advisor-content">
          <button 
            className="get-hint-btn"
            onClick={fetchSuggestion}
            disabled={loading}
          >
            {loading ? '⏳ Thinking...' : '🎯 Get AI Hint'}
          </button>
        </div>
      )}

      {activeTab === 'learn' && stats && (
        <div className="advisor-content">
          <div className="stats-box">
            <h4>📈 Training Progress</h4>
            <div className="stat-item">
              <span className="stat-label">Episodes Learned:</span>
              <span className="stat-value">{stats.total_training_episodes}</span>
            </div>
            
            <h4 style={{ marginTop: '1rem' }}>🎓 Best Strategies Found</h4>
            {Object.entries(stats.learned_strategies).map(([difficulty, strategy]) => (
              <div key={difficulty} className="strategy-item">
                <span className="difficulty-badge" data-level={difficulty}>
                  {difficulty.toUpperCase()}
                </span>
                {strategy.count > 0 ? (
                  <span className="strategy-actions">
                    {strategy.action_sequence.map(a => 
                      actionEmojis[a] ? `${actionEmojis[a]} ` : ''
                    )}
                  </span>
                ) : (
                  <span className="no-strategy">Learning in progress...</span>
                )}
              </div>
            ))}
            
            <div className="avg-rewards">
              <h4 style={{ marginTop: '1rem' }}>📊 Average Reward by Difficulty</h4>
              {Object.entries(stats.agent_stats.average_rewards_by_difficulty || {}).map(([diff, reward]) => (
                <div key={diff} className="reward-item">
                  <span>{diff}</span>
                  <div className="reward-bar">
                    <div 
                      className="reward-fill" 
                      style={{ width: `${reward * 100}%` }}
                    ></div>
                  </div>
                  <span>{reward.toFixed(2)}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
