/**
 * Diabetes Prediction Web App - Main JavaScript
 */

const API_BASE_URL = 'http://localhost:5001/api';

// Tab Navigation
document.addEventListener('DOMContentLoaded', function() {
    initTabs();
    initForm();
    loadModelInfo();
    loadAnalysis();
});

function initTabs() {
    const tabs = document.querySelectorAll('.nav-tab');
    const tabContents = document.querySelectorAll('.tab-content');

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const targetTab = tab.getAttribute('data-tab');

            // Remove active class from all tabs and contents
            tabs.forEach(t => t.classList.remove('active'));
            tabContents.forEach(tc => tc.classList.remove('active'));

            // Add active class to clicked tab and corresponding content
            tab.classList.add('active');
            document.getElementById(`${targetTab}-tab`).classList.add('active');
        });
    });
}

function initForm() {
    const form = document.getElementById('prediction-form');
    form.addEventListener('submit', handleFormSubmit);
}

async function handleFormSubmit(e) {
    e.preventDefault();
    
    const submitBtn = document.getElementById('predict-btn');
    const originalText = submitBtn.innerHTML;
    
    // Disable button and show loading
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Đang xử lý...';
    
    try {
        // Get form data
        const formData = new FormData(e.target);
        const data = {};
        
        // Convert form data to object
        for (let [key, value] of formData.entries()) {
            data[key] = parseFloat(value) || parseInt(value) || value;
        }
        
        // Make API call
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        
        const result = await response.json();
        
        if (response.ok) {
            displayResults(result);
        } else {
            showError(result.error || 'Có lỗi xảy ra khi dự đoán');
        }
    } catch (error) {
        showError('Không thể kết nối đến server. Vui lòng kiểm tra lại.');
        console.error('Error:', error);
    } finally {
        submitBtn.disabled = false;
        submitBtn.innerHTML = originalText;
    }
}

function displayResults(result) {
    const resultsSection = document.getElementById('results-section');
    const resultsContent = document.getElementById('results-content');
    
    const isRisk = result.prediction === 1;
    const probability = (result.probability * 100).toFixed(1);
    const confidence = (result.confidence * 100).toFixed(1);
    
    resultsContent.innerHTML = `
        <div class="result-box ${isRisk ? 'danger' : 'success'}">
            <h3>
                <i class="fas ${isRisk ? 'fa-exclamation-triangle' : 'fa-check-circle'}"></i>
                ${result.result_text}
            </h3>
            <p style="font-size: 1.1rem; margin: 15px 0;">
                Xác suất: <strong>${probability}%</strong>
            </p>
            <div class="probability-bar">
                <div class="probability-fill" style="width: ${probability}%">
                    ${probability}%
                </div>
            </div>
            <p style="margin-top: 15px; opacity: 0.9;">
                Độ tin cậy: ${confidence}%
            </p>
        </div>
        
        <div class="recommendations">
            <h4><i class="fas fa-lightbulb"></i> Khuyến Nghị</h4>
            <ul>
                ${result.recommendations.map(rec => `<li>${rec}</li>`).join('')}
            </ul>
        </div>
    `;
    
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

function showError(message) {
    const resultsSection = document.getElementById('results-section');
    const resultsContent = document.getElementById('results-content');
    
    resultsContent.innerHTML = `
        <div class="result-box danger">
            <h3><i class="fas fa-times-circle"></i> Lỗi</h3>
            <p>${message}</p>
        </div>
    `;
    
    resultsSection.style.display = 'block';
}

async function loadModelInfo() {
    try {
        const response = await fetch(`${API_BASE_URL}/model-info`);
        const data = await response.json();
        
        if (response.ok) {
            const modelInfoDiv = document.getElementById('model-info');
            modelInfoDiv.innerHTML = `
                <div class="model-info-item">
                    <span><strong>Model:</strong></span>
                    <span>${data.model_name.toUpperCase()}</span>
                </div>
                <div class="model-info-item">
                    <span><strong>Độ chính xác:</strong></span>
                    <span>${(data.accuracy * 100).toFixed(2)}%</span>
                </div>
                <div class="model-info-item">
                    <span><strong>F1-Score:</strong></span>
                    <span>${data.f1_score.toFixed(4)}</span>
                </div>
                <div class="model-info-item">
                    <span><strong>Ngưỡng tối ưu:</strong></span>
                    <span>${data.optimal_threshold.toFixed(4)}</span>
                </div>
                <div class="model-info-item">
                    <span><strong>Số features:</strong></span>
                    <span>${data.selected_features_count} / ${data.total_features}</span>
                </div>
            `;
        }
    } catch (error) {
        console.error('Error loading model info:', error);
        document.getElementById('model-info').innerHTML = 
            '<p style="color: var(--danger-color);">Không thể tải thông tin model</p>';
    }
}

async function loadAnalysis() {
    try {
        const response = await fetch(`${API_BASE_URL}/analyze`);
        const data = await response.json();
        
        if (response.ok) {
            displayAnalysis(data);
        }
    } catch (error) {
        console.error('Error loading analysis:', error);
        document.getElementById('analysis-content').innerHTML = 
            '<p style="color: var(--danger-color);">Không thể tải dữ liệu phân tích</p>';
    }
}

function displayAnalysis(data) {
    const analysisContent = document.getElementById('analysis-content');
    
    analysisContent.innerHTML = `
        <div class="stats-grid">
            <div class="stat-card">
                <h3>${data.total_samples.toLocaleString()}</h3>
                <p><i class="fas fa-users"></i> Tổng số mẫu</p>
            </div>
            <div class="stat-card">
                <h3>${data.diabetes_count.toLocaleString()}</h3>
                <p><i class="fas fa-heartbeat"></i> Số ca mắc bệnh</p>
            </div>
            <div class="stat-card">
                <h3>${data.diabetes_rate.toFixed(2)}%</h3>
                <p><i class="fas fa-percentage"></i> Tỷ lệ mắc bệnh</p>
            </div>
        </div>
        
        <div class="chart-container">
            <h3><i class="fas fa-chart-pie"></i> Phân Bố Dữ Liệu</h3>
            <div id="distribution-charts"></div>
        </div>
        
        <div class="chart-container">
            <h3><i class="fas fa-chart-line"></i> Tương Quan Với Bệnh Tiểu Đường</h3>
            <div id="correlation-chart"></div>
        </div>
        
        <div class="chart-container">
            <h3><i class="fas fa-table"></i> Thống Kê Features</h3>
            <div id="feature-stats"></div>
        </div>
    `;
    
    // Display distribution charts
    displayDistributionCharts(data.categorical_distribution);
    
    // Display correlation chart
    displayCorrelationChart(data.correlations);
    
    // Display feature statistics
    displayFeatureStats(data.feature_stats);
}

function displayDistributionCharts(distributions) {
    const container = document.getElementById('distribution-charts');
    let html = '';
    
    for (const [feature, dist] of Object.entries(distributions)) {
        const total = Object.values(dist).reduce((a, b) => a + b, 0);
        html += `
            <div style="margin-bottom: 30px;">
                <h4 style="color: var(--primary-color); margin-bottom: 15px;">${getFeatureName(feature)}</h4>
                <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                    ${Object.entries(dist).map(([key, value]) => {
                        const percentage = ((value / total) * 100).toFixed(1);
                        return `
                            <div style="flex: 1; min-width: 150px; background: var(--bg-color); padding: 15px; border-radius: 8px;">
                                <div style="font-weight: 600; color: var(--primary-color);">${getValueLabel(feature, key)}</div>
                                <div style="font-size: 1.5rem; margin: 10px 0;">${value.toLocaleString()}</div>
                                <div style="color: var(--text-secondary); font-size: 0.9rem;">${percentage}%</div>
                            </div>
                        `;
                    }).join('')}
                </div>
            </div>
        `;
    }
    
    container.innerHTML = html;
}

function displayCorrelationChart(correlations) {
    const container = document.getElementById('correlation-chart');
    const sorted = Object.entries(correlations).sort((a, b) => b[1] - a[1]);
    
    let html = '<div style="display: flex; flex-direction: column; gap: 15px;">';
    
    sorted.forEach(([feature, value]) => {
        const percentage = (value * 100).toFixed(1);
        html += `
            <div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                    <span><strong>${getFeatureName(feature)}</strong></span>
                    <span>${percentage}%</span>
                </div>
                <div style="background: var(--border-color); border-radius: 10px; height: 25px; overflow: hidden;">
                    <div style="background: linear-gradient(90deg, var(--primary-color), var(--secondary-color)); height: 100%; width: ${percentage}%; border-radius: 10px; transition: width 0.5s ease;"></div>
                </div>
            </div>
        `;
    });
    
    html += '</div>';
    container.innerHTML = html;
}

function displayFeatureStats(stats) {
    const container = document.getElementById('feature-stats');
    let html = '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px;">';
    
    for (const [feature, stat] of Object.entries(stats)) {
        html += `
            <div style="background: var(--bg-color); padding: 20px; border-radius: 12px;">
                <h4 style="color: var(--primary-color); margin-bottom: 15px;">${getFeatureName(feature)}</h4>
                <div style="display: flex; flex-direction: column; gap: 8px;">
                    <div style="display: flex; justify-content: space-between;">
                        <span>Trung bình:</span>
                        <strong>${stat.mean.toFixed(2)}</strong>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span>Trung vị:</span>
                        <strong>${stat.median.toFixed(2)}</strong>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span>Min:</span>
                        <strong>${stat.min.toFixed(2)}</strong>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span>Max:</span>
                        <strong>${stat.max.toFixed(2)}</strong>
                    </div>
                </div>
            </div>
        `;
    }
    
    html += '</div>';
    container.innerHTML = html;
}

function getFeatureName(feature) {
    const names = {
        'HighBP': 'Huyết Áp Cao',
        'HighChol': 'Cholesterol Cao',
        'BMI': 'BMI',
        'Smoker': 'Hút Thuốc',
        'PhysActivity': 'Hoạt Động Thể Chất',
        'Fruits': 'Ăn Trái Cây',
        'GenHlth': 'Sức Khỏe Tổng Quát',
        'MentHlth': 'Sức Khỏe Tinh Thần',
        'PhysHlth': 'Sức Khỏe Thể Chất',
        'Age': 'Tuổi',
        'Education': 'Học Vấn',
        'Income': 'Thu Nhập',
        'Sex': 'Giới Tính'
    };
    return names[feature] || feature;
}

function getValueLabel(feature, value) {
    const labels = {
        'HighBP': { '0': 'Không', '1': 'Có' },
        'HighChol': { '0': 'Không', '1': 'Có' },
        'Smoker': { '0': 'Không', '1': 'Có' },
        'PhysActivity': { '0': 'Không', '1': 'Có' },
        'Fruits': { '0': 'Không/Ít', '1': 'Có' },
        'Sex': { '0': 'Nữ', '1': 'Nam' },
        'GenHlth': { '1': 'Rất tốt', '2': 'Tốt', '3': 'Khá tốt', '4': 'Trung bình', '5': 'Kém' }
    };
    
    return labels[feature]?.[value] || value;
}


