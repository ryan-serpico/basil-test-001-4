import React from 'react';

const Header = ({ projectName, statistics }) => {
  return (
    <header className="app-header">
      <div className="header-content">
        <div className="header-title">
          <h1>🔍 {projectName}</h1>
          <p className="subtitle">Semantic Document Search</p>
        </div>

        {statistics && (
          <div className="header-stats">
            <div className="stat-item">
              <span className="stat-value">{statistics.total_documents}</span>
              <span className="stat-label">Documents</span>
            </div>
            <div className="stat-item">
              <span className="stat-value">{statistics.total_chunks}</span>
              <span className="stat-label">Chunks</span>
            </div>
            <div className="stat-item">
              <span className="stat-value">{statistics.total_tokens?.toLocaleString()}</span>
              <span className="stat-label">Tokens</span>
            </div>
          </div>
        )}
      </div>
    </header>
  );
};

export default Header;