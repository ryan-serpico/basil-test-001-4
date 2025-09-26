import React, { useState } from 'react';

const SearchResults = ({ results, loading }) => {
  const [expandedResults, setExpandedResults] = useState(new Set());

  const toggleExpanded = (index) => {
    const newExpanded = new Set(expandedResults);
    if (newExpanded.has(index)) {
      newExpanded.delete(index);
    } else {
      newExpanded.add(index);
    }
    setExpandedResults(newExpanded);
  };

  const getSimilarityColor = (score) => {
    // Color coding for similarity scores
    if (score >= 0.8) return '#22c55e'; // Green - High similarity
    if (score >= 0.6) return '#eab308'; // Yellow - Medium similarity
    return '#ef4444'; // Red - Low similarity
  };

  const getSimilarityLabel = (score) => {
    if (score >= 0.8) return 'High';
    if (score >= 0.6) return 'Medium';
    return 'Low';
  };

  if (loading) {
    return (
      <div className="search-results">
        <div className="loading-results">
          <div className="loading-spinner small"></div>
          <p>Searching documents...</p>
        </div>
      </div>
    );
  }

  if (results.length === 0) {
    return null;
  }

  return (
    <div className="search-results">
      {results.map((result, index) => (
        <div key={index} className="result-item">
          <div className="result-header">
            <div className="result-meta">
              <span className="source-document">
                📄 {result.source_document}
              </span>
              <span
                className="similarity-score"
                style={{ color: getSimilarityColor(result.similarity_score) }}
              >
                {getSimilarityLabel(result.similarity_score)} ({(result.similarity_score * 100).toFixed(1)}%)
              </span>
              <span className="chunk-position">
                Chunk {result.chunk_position + 1}
              </span>
            </div>
            <button
              onClick={() => toggleExpanded(index)}
              className="expand-button"
            >
              {expandedResults.has(index) ? '▼ Less' : '▶ More'}
            </button>
          </div>

          <div className="result-content">
            <div className="chunk-text">
              {result.chunk_text}
            </div>

            {expandedResults.has(index) && (
              <div className="expanded-content">
                {result.context_before && (
                  <div className="context-section">
                    <h4>Context Before:</h4>
                    <div className="context-text context-before">
                      {result.context_before}
                    </div>
                  </div>
                )}

                {result.context_after && (
                  <div className="context-section">
                    <h4>Context After:</h4>
                    <div className="context-text context-after">
                      {result.context_after}
                    </div>
                  </div>
                )}

                <div className="result-details">
                  <div className="detail-item">
                    <strong>Document:</strong> {result.source_document}
                  </div>
                  <div className="detail-item">
                    <strong>Position:</strong> Chunk {result.chunk_position + 1}
                  </div>
                  <div className="detail-item">
                    <strong>Similarity:</strong> {(result.similarity_score * 100).toFixed(2)}%
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      ))}
    </div>
  );
};

export default SearchResults;