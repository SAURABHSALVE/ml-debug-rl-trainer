import { useState, useEffect } from 'react';
import './LearningDashboard.css';

export default function LearningDashboard({ isOpen, onClose }) {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(false);
  const [selectedDifficulty, setSelectedDifficulty] = useState('easy');

  useEffect(() => {
    if (isOpen) {
      fetchStats();
      // Refresh stats every 5 seconds
      const interval = setInterval(fetchStats, 5000);
      return () => clearInterval(interval);
    }
  }, [isOpen]);

  const fetchStats = async () => {
    setLoading(true);
    try {
      const res = await fetch('/api/agent-stats');
      if (res.ok) {
        const data = await res.json();
        setStats(data);
      }
    } catch (e) {
      console.error('Failed to fetch stats:', e);
    } finally {
      setLoading(false);
    }
  };

  if (!isOpen) return null;

  const agentStats = stats?.agent_stats || {};
  const strategies = stats?.learned_strategies || {};
  const totalEpisodes = stats?.total_training_episodes || 0;

  const getEpisodeHealth = (episodes) => {
    if (episodes < 100) return { status: 'training', color: '#f97316' };
    if (episodes < 300) return { status: 'learning', color: '#eab308' };
    if (episodes < 500) return { status: 'confident', color: '#84cc16' };
    return { status: 'expert', color: '#22c55e' };
  };

  const health = getEpisodeHealth(totalEpisodes);
  const selectedStrategy = strategies[selectedDifficulty];

  const actionEmojis = {
    fetch_logs: '📋',
    fetch_config: '⚙️',
    fetch_loss_curve: '📈',
    fetch_diagnostics: '🔍',
    fetch_class_data: '📊',
    diagnose: '🎯',
  };

  return (
    <div className="dashboard-overlay" onClick={onClose}>
      <div className="dashboard-panel" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className="dashboard-header">
          <h2>🧠 Agent Learning Dashboard</h2>
          <button className="close-btn" onClick={onClose}>✕</button>
        </div>

        {/* Health Status */}
        <div className="health-section">
          <div className="health-badge" style={{ backgroundColor: health.color }}>
            {health.status.toUpperCase()}
          </div>
          <div className="health-info">
            <p className="episodes-count">{totalEpisodes} episodes learned</p>
            <p className="health-description">
              {totalEpisodes < 100
                ? '🌱 Agent is exploring, strategies not yet stable'
                : totalEpisodes < 300
                ? '🌿 Agent learning good strategies'
                : totalEpisodes < 500
                ? '🌳 Agent confident, high consistency'
                : '🏆 Agent expert, near-optimal strategies'}
            </p>
          </div>
        </div>

        {/* Tabs for Difficulty */}
        <div className="difficulty-tabs">
          {['easy', 'medium', 'hard'].map((diff) => (
            <button
              key={diff}
              className={`diff-tab ${selectedDifficulty === diff ? 'active' : ''}`}
              onClick={() => setSelectedDifficulty(diff)}
              data-level={diff}
            >
              <span className="diff-name">{diff.toUpperCase()}</span>
              {agentStats.average_rewards_by_difficulty?.[diff] && (
                <span className="diff-score">
                  {(agentStats.average_rewards_by_difficulty[diff] * 100).toFixed(0)}%
                </span>
              )}
            </button>
          ))}
        </div>

        {/* Strategy Display */}
        {selectedStrategy && (
          <div className="strategy-section">
            <h3>Learned Strategy for {selectedDifficulty.toUpperCase()}</h3>
            {selectedStrategy.count > 0 ? (
              <div className="strategy-flow">
                {selectedStrategy.action_sequence.map((action, idx) => (
                  <div key={idx} className="strategy-step">
                    <div className="step-number">{idx + 1}</div>
                    <div className="action-name">
                      <span className="emoji">{actionEmojis[action]}</span>
                      <span>{action.replace('_', ' ')}</span>
                    </div>
                    {idx < selectedStrategy.action_sequence.length - 1 && (
                      <div className="arrow">→</div>
                    )}
                  </div>
                ))}
                <div className="strategy-step final">
                  <div className="step-number">✓</div>
                  <div className="action-name">
                    <span className="emoji">🎯</span>
                    <span>diagnose</span>
                  </div>
                </div>
              </div>
            ) : (
              <p className="no-strategy">
                Agent still exploring. Train for more episodes to discover a stable strategy.
              </p>
            )}
          </div>
        )}

        {/* Performance Metrics */}
        <div className="metrics-section">
          <h3>Performance Metrics (Last 100 Episodes)</h3>
          <div className="metrics-grid">
            {Object.entries(agentStats.average_rewards_by_difficulty || {}).map(
              ([difficulty, reward]) => (
                <div key={difficulty} className="metric-card">
                  <div className="metric-label">{difficulty}</div>
                  <div className="metric-bar">
                    <div
                      className="metric-fill"
                      style={{
                        width: `${reward * 100}%`,
                        backgroundColor:
                          reward > 0.8
                            ? '#22c55e'
                            : reward > 0.6
                            ? '#84cc16'
                            : reward > 0.4
                            ? '#eab308'
                            : '#f97316',
                      }}
                    ></div>
                  </div>
                  <div className="metric-value">{reward.toFixed(2)}</div>
                </div>
              )
            )}
          </div>
        </div>

        {/* Learning Curve */}
        <div className="curve-section">
          <h3>Learning Trajectory</h3>
          <div className="learning-curve">
            <svg viewBox="0 0 300 150" preserveAspectRatio="xMidYMid meet">
              {/* Grid */}
              <line x1="30" y1="130" x2="280" y2="130" stroke="#e5e7eb" strokeWidth="1" />
              <line x1="30" y1="20" x2="30" y2="130" stroke="#e5e7eb" strokeWidth="1" />
              
              {/* Axes labels */}
              <text x="295" y="135" fontSize="12" fill="#6b7280">Episodes</text>
              <text x="5" y="15" fontSize="12" fill="#6b7280">Reward</text>
              
              {/* Simulated learning curve */}
              <path
                d="M 30,125 Q 80,115 130,90 T 230,40"
                stroke="#8b5cf6"
                strokeWidth="2"
                fill="none"
              />
              
              {/* Agent status marker */}
              <circle
                cx={30 + (totalEpisodes / 500) * 250}
                cy={130 - (getRewardForEpisodes(totalEpisodes) * 100)}
                r="4"
                fill="#8b5cf6"
              />
            </svg>
            <p className="curve-label">
              Current position: {totalEpisodes} episodes
            </p>
          </div>
        </div>

        {/* Action Recommendations */}
        <div className="recommendations-section">
          <h3>What This Means For You</h3>
          {totalEpisodes < 100 ? (
            <div className="advice-box">
              <p>🌱 <strong>Agent is still exploring.</strong></p>
              <p>Recommendations might be random. Keep playing episodes to help it learn patterns.</p>
            </div>
          ) : totalEpisodes < 300 ? (
            <div className="advice-box">
              <p>🌿 <strong>Agent found some good patterns.</strong></p>
              <p>Recommendations are better than random. Start noticing its preferences.</p>
            </div>
          ) : totalEpisodes < 500 ? (
            <div className="advice-box">
              <p>🌳 <strong>Agent has solid strategies.</strong></p>
              <p>Follow its recommendations for best results. It's learned task-specific patterns.</p>
            </div>
          ) : (
            <div className="advice-box">
              <p>🏆 <strong>Agent is an expert debugger.</strong></p>
              <p>Its strategies solve {selectedDifficulty} task ~90% of the time. Follow it for optimal results!</p>
            </div>
          )}
        </div>

        {/* Statistics Grid */}
        <div className="stats-grid">
          <div className="stat-box">
            <span className="stat-icon">📚</span>
            <span className="stat-value">{totalEpisodes}</span>
            <span className="stat-label">Episodes</span>
          </div>
          <div className="stat-box">
            <span className="stat-icon">✓</span>
            <span className="stat-value">{agentStats.strategies_learned ? 'Yes' : 'No'}</span>
            <span className="stat-label">Learned</span>
          </div>
          <div className="stat-box">
            <span className="stat-icon">📈</span>
            <span className="stat-value">
              {agentStats.average_rewards_by_difficulty?.easy
                ? (agentStats.average_rewards_by_difficulty.easy * 100).toFixed(0)
                : '0'}
              %
            </span>
            <span className="stat-label">Easy Task</span>
          </div>
        </div>

        {/* Footer */}
        <div className="dashboard-footer">
          <p>💡 Run `python train_agent.py --episodes 1000` to train offline</p>
          <button className="refresh-btn" onClick={fetchStats} disabled={loading}>
            {loading ? '⏳ Refreshing...' : '↻ Refresh'}
          </button>
        </div>
      </div>
    </div>
  );
}

// Helper function to estimate reward based on episodes
function getRewardForEpisodes(episodes) {
  // Simplified learning curve: sigmoid-like growth
  const progress = episodes / 500;
  return 0.2 + (0.6 * progress) / (1 + progress);
}
