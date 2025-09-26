import React, { useState } from 'react';

const Statistics = ({ statistics, documents }) => {
  const [showDocuments, setShowDocuments] = useState(false);

  if (!statistics) {
    return null;
  }

  const formatBytes = (bytes) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="statistics-panel">
      <h2>Collection Statistics</h2>

      <div className="stats-grid">
        <div className="stat-card">
          <h3>📚 Documents</h3>
          <div className="stat-number">{statistics.total_documents}</div>
          <div className="stat-detail">
            Avg size: {formatBytes(statistics.average_document_size)}
          </div>
        </div>

        <div className="stat-card">
          <h3>🧩 Text Chunks</h3>
          <div className="stat-number">{statistics.total_chunks}</div>
          <div className="stat-detail">
            Avg per doc: {statistics.average_chunks_per_document}
          </div>
        </div>

        <div className="stat-card">
          <h3>🔤 Tokens</h3>
          <div className="stat-number">{statistics.total_tokens?.toLocaleString()}</div>
          <div className="stat-detail">
            Text processed
          </div>
        </div>

        <div className="stat-card">
          <h3>💾 Database</h3>
          <div className="stat-number">
            {statistics.collection_loaded ? '✅ Loaded' : '❌ Not Loaded'}
          </div>
          <div className="stat-detail">
            {statistics.chroma_collection_count} vectors
          </div>
        </div>
      </div>

      {documents && documents.length > 0 && (
        <div className="documents-section">
          <div className="documents-header">
            <h3>📄 Document List</h3>
            <button
              onClick={() => setShowDocuments(!showDocuments)}
              className="toggle-button"
            >
              {showDocuments ? '▼ Hide' : '▶ Show'} ({documents.length} documents)
            </button>
          </div>

          {showDocuments && (
            <div className="documents-list">
              {documents.map((doc, index) => (
                <div key={index} className="document-item">
                  <div className="document-name">{doc.name}</div>
                  <div className="document-stats">
                    <span>{doc.chunks} chunks</span>
                    <span>{doc.tokens?.toLocaleString()} tokens</span>
                    <span>{formatBytes(doc.size_bytes)}</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      <div className="stats-footer">
        <small>
          Last updated: {new Date().toLocaleString()}
        </small>
      </div>
    </div>
  );
};

export default Statistics;