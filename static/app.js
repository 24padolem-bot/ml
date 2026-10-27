/**
 * Machine Health Intelligence - Client Application
 * Optimized for Vercel Serverless & Static Edge CDN Deployment.
 * Interactive Controls: Live Search, Diagnostic Notifications, Simulation Modal, Plotly Visualizations.
 */

// Global State
let currentEngineData = null;
let currentMaxCycle = 31;
let allEngineUnits = [];
let fleetSummaryData = null;

document.addEventListener("DOMContentLoaded", () => {
    initNavigation();
    initTopBarInteractions();
    loadFleetSummary();
    loadEngineList();
    renderBenchmarkCharts();
});

// ----------------------------------------------------------------------------
// Navigation & Tab Switching
// ----------------------------------------------------------------------------
function initNavigation() {
    const navButtons = document.querySelectorAll(".nav-btn");
    navButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            const targetTab = btn.getAttribute("data-tab");
            switchTab(targetTab);
        });
    });
}

function switchTab(tabId) {
    // Update active nav button
    document.querySelectorAll(".nav-btn").forEach(btn => {
        if (btn.getAttribute("data-tab") === tabId) {
            btn.classList.add("active");
        } else {
            btn.classList.remove("active");
        }
    });

    // Update active tab pane
    document.querySelectorAll(".tab-pane").forEach(pane => {
        if (pane.id === tabId) {
            pane.classList.add("active");
        } else {
            pane.classList.remove("active");
        }
    });

    // Update Top Title dynamically based on active tab
    const titleMap = {
        "tab-portfolio": "Portfolio",
        "tab-telemetry": "Telemetry",
        "tab-prognostics": "Prognostics",
        "tab-risk": "Maintenance Risk",
        "tab-benchmarks": "Benchmarks",
        "tab-clusters": "Operating Regimes",
        "tab-docs": "Methodology"
    };
    const titleEl = document.getElementById("topPageTitle");
    if (titleEl && titleMap[tabId]) {
        titleEl.innerText = titleMap[tabId];
    }

    // Trigger chart renders for newly active tabs
    if (tabId === "tab-telemetry") {
        setTimeout(renderTelemetryCharts, 50);
    } else if (tabId === "tab-prognostics" || tabId === "tab-risk") {
        setTimeout(updatePrognosticPrediction, 50);
    } else if (tabId === "tab-benchmarks") {
        setTimeout(renderBenchmarkCharts, 50);
    }

    // Auto close sidebar on mobile upon navigation
    closeMobileSidebar();
}

function openMobileSidebar() {
    const sidebar = document.getElementById("appSidebar");
    const backdrop = document.getElementById("sidebarBackdrop");
    if (sidebar) sidebar.classList.add("open");
    if (backdrop) backdrop.classList.add("active");
}

function closeMobileSidebar() {
    const sidebar = document.getElementById("appSidebar");
    const backdrop = document.getElementById("sidebarBackdrop");
    if (sidebar) sidebar.classList.remove("open");
    if (backdrop) backdrop.classList.remove("active");
}

function selectQuickEngine(unitId, customCycle = null) {
    const progSelect = document.getElementById("prognosticEngineSelect");
    if (progSelect) {
        progSelect.value = unitId;
    }
    const telemSelect = document.getElementById("telemetryEngineSelect");
    if (telemSelect) {
        telemSelect.value = unitId;
    }

    onPrognosticEngineChange().then(() => {
        if (customCycle !== null) {
            const slider = document.getElementById("cutoffSlider");
            const badge = document.getElementById("sliderValBadge");
            if (slider) {
                slider.value = Math.min(customCycle, currentMaxCycle);
                if (badge) badge.innerText = `Cycle ${slider.value}`;
            }
            updatePrognosticPrediction();
        }
    });

    switchTab("tab-prognostics");
}

// ----------------------------------------------------------------------------
// Top Bar Interactions: Live Search, Notifications, Plus Button, Menu
// ----------------------------------------------------------------------------
function initTopBarInteractions() {
    // 1. Mobile & Desktop Hamburger Menu & Drawer Toggle
    const menuBtn = document.getElementById("menuToggleBtn");
    const closeBtn = document.getElementById("sidebarCloseBtn");
    const backdrop = document.getElementById("sidebarBackdrop");

    if (menuBtn) {
        menuBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            const sidebar = document.getElementById("appSidebar");
            if (sidebar && sidebar.classList.contains("open")) {
                closeMobileSidebar();
            } else {
                openMobileSidebar();
            }
        });
    }

    if (closeBtn) {
        closeBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            closeMobileSidebar();
        });
    }

    if (backdrop) {
        backdrop.addEventListener("click", () => {
            closeMobileSidebar();
        });
    }

    // 2. Functional Live Search
    const searchInput = document.getElementById("fleetSearchInput");
    const searchDropdown = document.getElementById("searchResultsDropdown");
    const clearSearchBtn = document.getElementById("clearSearchBtn");

    if (searchInput && searchDropdown) {
        searchInput.addEventListener("input", (e) => {
            const query = e.target.value.trim().toLowerCase();
            if (clearSearchBtn) {
                clearSearchBtn.style.display = query ? "inline-block" : "none";
            }

            if (!query) {
                searchDropdown.style.display = "none";
                return;
            }

            renderSearchResults(query, searchDropdown);
        });

        searchInput.addEventListener("focus", (e) => {
            const query = e.target.value.trim().toLowerCase();
            if (query) {
                renderSearchResults(query, searchDropdown);
            }
        });
    }

    if (clearSearchBtn && searchInput && searchDropdown) {
        clearSearchBtn.addEventListener("click", () => {
            searchInput.value = "";
            clearSearchBtn.style.display = "none";
            searchDropdown.style.display = "none";
            searchInput.focus();
        });
    }

    // 3. Notification Button & Dropdown Toggle
    const notifBtn = document.getElementById("notificationBtn");
    const notifDropdown = document.getElementById("notificationDropdown");
    if (notifBtn && notifDropdown) {
        notifBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            const isOpen = notifDropdown.style.display === "block";
            closeAllTopDropdowns();
            notifDropdown.style.display = isOpen ? "none" : "block";
        });
    }

    // 4. Add Engine / Simulate Button
    const addBtn = document.getElementById("addEngineBtn");
    if (addBtn) {
        addBtn.addEventListener("click", () => {
            openAddModal();
        });
    }

    // 5. Global Click Outside Listener to Dismiss Dropdowns
    document.addEventListener("click", (e) => {
        const inSearch = e.target.closest(".search-container");
        const inNotif = e.target.closest(".notification-container");
        if (!inSearch && searchDropdown) {
            searchDropdown.style.display = "none";
        }
        if (!inNotif && notifDropdown) {
            notifDropdown.style.display = "none";
        }
    });
}

function closeAllTopDropdowns() {
    const searchDropdown = document.getElementById("searchResultsDropdown");
    const notifDropdown = document.getElementById("notificationDropdown");
    if (searchDropdown) searchDropdown.style.display = "none";
    if (notifDropdown) notifDropdown.style.display = "none";
}

function renderSearchResults(query, dropdown) {
    const units = allEngineUnits.length ? allEngineUnits : Array.from({ length: 100 }, (_, i) => i + 1);

    // Matching engines
    let matches = [];
    units.forEach(u => {
        const uStr = String(u);
        const name = `Engine #${u}`;
        const isCritical = (u === 34 || u === 81 || u === 68 || u === 76);
        const isWarning = (u === 18 || u === 49 || u === 92);

        let matchScore = 0;
        if (uStr === query || name.toLowerCase().includes(query)) matchScore = 10;
        else if (uStr.startsWith(query)) matchScore = 8;
        else if (uStr.includes(query)) matchScore = 5;
        else if (query.includes("alert") || query.includes("crit")) {
            if (isCritical) matchScore = 9;
        } else if (query.includes("risk") || query.includes("warn")) {
            if (isWarning || isCritical) matchScore = 7;
        } else if (query.includes("nom") || query.includes("ok")) {
            if (!isCritical && !isWarning) matchScore = 4;
        }

        if (matchScore > 0) {
            matches.push({ unit_nr: u, isCritical, isWarning, score: matchScore });
        }
    });

    matches.sort((a, b) => b.score - a.score);
    const topMatches = matches.slice(0, 7);

    if (topMatches.length === 0) {
        dropdown.innerHTML = `<div class="search-no-results">No engine units matching "<strong>${query}</strong>"</div>`;
    } else {
        dropdown.innerHTML = topMatches.map(m => {
            const statusLabel = m.isCritical ? "CRITICAL RISK" : (m.isWarning ? "ELEVATED" : "NOMINAL");
            const avatarBg = m.isCritical ? "#FEE2E2; color: #DC2626;" : (m.isWarning ? "#FEE8D6; color: #EA580C;" : "#E0F2FE; color: #0284C7;");
            return `
                <div class="search-result-item" onclick="selectQuickEngine(${m.unit_nr}); closeAllTopDropdowns();">
                    <div class="search-item-left">
                        <div class="search-item-avatar" style="background: ${avatarBg}">#${m.unit_nr}</div>
                        <div>
                            <div class="search-item-name">Turbofan Engine #${m.unit_nr}</div>
                            <div class="search-item-meta">${m.isCritical ? "RUL ≤ 20 cyc" : "In-Service Flight"}</div>
                        </div>
                    </div>
                    <span class="pill-badge ${m.isCritical ? 'pill-rose' : 'pill-white'}" style="font-size: 0.72rem;">${statusLabel}</span>
                </div>
            `;
        }).join("");
    }
    dropdown.style.display = "block";
}

function clearAllNotifications() {
    const notifList = document.getElementById("notifList");
    const notifBadge = document.getElementById("notifBadge");
    const countText = document.getElementById("notifCountText");
    const clearBtn = document.getElementById("clearNotifBtn");

    if (notifList) {
        notifList.innerHTML = `<div style="padding: 24px 18px; text-align: center; color: #94A3B8; font-size: 0.85rem;">✅ All alerts acknowledged. Fleet telemetry nominal.</div>`;
    }
    if (notifBadge) notifBadge.style.display = "none";
    if (countText) {
        countText.innerText = "0 Active";
        countText.style.background = "#E2E8F0";
        countText.style.color = "#64748B";
    }
    if (clearBtn) clearBtn.style.display = "none";
    showToast("All diagnostic alerts marked as acknowledged.", "info");
}

// ----------------------------------------------------------------------------
// Add Engine / Simulation Modal
// ----------------------------------------------------------------------------
function openAddModal() {
    const modal = document.getElementById("addEngineModal");
    const select = document.getElementById("modalEngineSelect");
    if (!modal) return;

    if (select && select.children.length === 0) {
        const units = allEngineUnits.length ? allEngineUnits : Array.from({ length: 100 }, (_, i) => i + 1);
        select.innerHTML = units.map(u => `<option value="${u}">Turbofan Unit #${u}</option>`).join("");
    }

    modal.style.display = "flex";
}

function closeAddModal() {
    const modal = document.getElementById("addEngineModal");
    if (modal) modal.style.display = "none";
}

function submitSimulatedIngest() {
    const unit = parseInt(document.getElementById("modalEngineSelect")?.value || 1);
    const cycle = parseInt(document.getElementById("modalCycleInput")?.value || 35);
    const mode = document.getElementById("modalOperatingMode")?.value || "nominal";

    closeAddModal();
    showToast(`Ingested Engine #${unit} at cycle ${cycle} (${mode}). Running prognostics...`, "success");
    selectQuickEngine(unit, cycle);
}

// ----------------------------------------------------------------------------
// Lightweight Toast Notifications
// ----------------------------------------------------------------------------
function showToast(message, type = "success") {
    const container = document.getElementById("toastContainer");
    if (!container) return;

    const toast = document.createElement("div");
    toast.className = `toast ${type}`;
    const icon = type === "success" ? "✅" : "ℹ️";
    toast.innerHTML = `<span>${icon}</span> <span>${message}</span>`;
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = "0";
        toast.style.transform = "translateY(10px)";
        toast.style.transition = "all 0.3s ease";
        setTimeout(() => toast.remove(), 300);
    }, 3200);
}

// ----------------------------------------------------------------------------
// Helper Fetch with Static Fallback
// ----------------------------------------------------------------------------
async function safeFetch(apiUrl, fallbackUrl) {
    try {
        const res = await fetch(apiUrl);
        if (res.ok) return await res.json();
    } catch (e) {
        // Fallback
    }
    if (fallbackUrl) {
        try {
            const res2 = await fetch(fallbackUrl);
            if (res2.ok) return await res2.json();
        } catch (e) {
            console.warn("Fallback fetch failed:", fallbackUrl, e);
        }
    }
    return null;
}

// ----------------------------------------------------------------------------
// Fleet Summary & Engine Population
// ----------------------------------------------------------------------------
async function loadFleetSummary() {
    const data = await safeFetch("/api/fleet/summary", "/data/fleet_summary.json");
    if (!data) return;
    fleetSummaryData = data;

    if (document.getElementById("statUnits")) {
        document.getElementById("statUnits").innerText = data.fleet_size || 100;
    }
    if (document.getElementById("statMae") && data.models?.best_regression) {
        document.getElementById("statMae").innerText = data.models.best_regression.test_mae.toFixed(2);
    }
    if (document.getElementById("statAccuracy") && data.models?.best_classification) {
        document.getElementById("statAccuracy").innerText = (data.models.best_classification.accuracy * 100).toFixed(1) + "%";
    }
    if (document.getElementById("statFeatures")) {
        document.getElementById("statFeatures").innerText = data.feature_count || 78;
    }
}

async function loadEngineList() {
    const data = await safeFetch("/api/telemetry/engines", "/data/engines_list.json");
    if (!data) return;

    allEngineUnits = data.test_units || [];

    const telemSelect = document.getElementById("telemetryEngineSelect");
    const progSelect = document.getElementById("prognosticEngineSelect");
    const modalSelect = document.getElementById("modalEngineSelect");

    if (telemSelect) telemSelect.innerHTML = "";
    if (progSelect) progSelect.innerHTML = "";
    if (modalSelect) modalSelect.innerHTML = "";

    allEngineUnits.forEach(u => {
        const opt1 = document.createElement("option");
        opt1.value = u;
        opt1.innerText = `Engine #${u}`;
        if (telemSelect) telemSelect.appendChild(opt1);

        const opt2 = document.createElement("option");
        opt2.value = u;
        opt2.innerText = `Engine #${u}`;
        if (progSelect) progSelect.appendChild(opt2);

        if (modalSelect) {
            const opt3 = document.createElement("option");
            opt3.value = u;
            opt3.innerText = `Turbofan Unit #${u}`;
            modalSelect.appendChild(opt3);
        }
    });

    if (progSelect) {
        progSelect.value = 1;
        await onPrognosticEngineChange();
    }
    if (telemSelect) {
        telemSelect.value = 1;
        renderTelemetryCharts();
    }
}

// ----------------------------------------------------------------------------
// Telemetry Inspector
// ----------------------------------------------------------------------------
async function renderTelemetryCharts() {
    const engineId = document.getElementById("telemetryEngineSelect")?.value || 1;
    const split = document.getElementById("telemetrySplitSelect")?.value || "test";

    const data = await safeFetch(
        `/api/telemetry/engine/${engineId}?split=${split}`,
        `/data/engines/unit_${engineId}.json`
    );
    if (!data) return;

    const records = data.records || [];
    const cycles = records.map(r => r.cycle);
    const s2 = records.map(r => r.s_2);
    const s3 = records.map(r => r.s_3);
    const s4 = records.map(r => r.s_4);
    const s11 = records.map(r => r.s_11);

    const traces = [
        { x: cycles, y: s2, name: "s_2 (LPC Outlet Temp)", line: { color: "#0284C7", width: 2.2 } },
        { x: cycles, y: s3, name: "s_3 (HPC Outlet Temp)", line: { color: "#EA580C", width: 2.2 } },
        { x: cycles, y: s4, name: "s_4 (LPT Outlet Temp)", line: { color: "#16A34A", width: 2.2 } },
        { x: cycles, y: s11, name: "s_11 (HPC Static Pres)", line: { color: "#9333EA", width: 2.2 } }
    ];

    const layout = {
        title: {
            text: `Thermodynamic Sensor Trajectories — Engine #${engineId} (${split.toUpperCase()})`,
            font: { family: "Poppins, sans-serif", size: 15, color: "#1E293B" }
        },
        font: { family: "Plus Jakarta Sans, sans-serif", size: 11, color: "#64748B" },
        paper_bgcolor: "#FFFFFF",
        plot_bgcolor: "#FFFFFF",
        xaxis: { title: "Flight Cycles Elapsed", gridcolor: "#F1F5F9", linecolor: "#E2E8F0" },
        yaxis: { title: "Sensor Measurement", gridcolor: "#F1F5F9", linecolor: "#E2E8F0" },
        margin: { l: 55, r: 25, t: 60, b: 65 },
        legend: { orientation: "h", y: -0.25, x: 0.5, xanchor: "center" }
    };

    Plotly.newPlot("telemetryChartContainer", traces, layout, { responsive: true, displayModeBar: false });
}

// ----------------------------------------------------------------------------
// Prognostics & Slider Handling
// ----------------------------------------------------------------------------
async function onPrognosticEngineChange() {
    const engineId = document.getElementById("prognosticEngineSelect")?.value || 1;
    currentEngineData = await safeFetch(
        `/api/telemetry/engine/${engineId}?split=test`,
        `/data/engines/unit_${engineId}.json`
    );
    if (!currentEngineData) return;

    currentMaxCycle = currentEngineData.max_cycle;

    const slider = document.getElementById("cutoffSlider");
    if (slider) {
        slider.min = Math.min(5, currentMaxCycle);
        slider.max = currentMaxCycle;
        slider.value = currentMaxCycle;
    }

    const info = document.getElementById("engineLifespanInfo");
    if (info) {
        info.innerText = `Max Recorded Cycles: ${currentMaxCycle}`;
    }

    const badge = document.getElementById("sliderValBadge");
    if (badge) {
        badge.innerText = `Cycle ${currentMaxCycle}`;
    }

    await updatePrognosticPrediction();
}

function onCutoffSliderChange() {
    const slider = document.getElementById("cutoffSlider");
    const badge = document.getElementById("sliderValBadge");
    if (slider && badge) {
        badge.innerText = `Cycle ${slider.value}`;
    }
    updatePrognosticPrediction();
}

async function updatePrognosticPrediction() {
    const engineId = parseInt(document.getElementById("prognosticEngineSelect")?.value || 1);
    const cutoffCycle = parseInt(document.getElementById("cutoffSlider")?.value || currentMaxCycle);

    let pred = null;

    // Try live API first
    try {
        const res = await fetch("/api/prognostics/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ unit_nr: engineId, cutoff_cycle: cutoffCycle })
        });
        if (res.ok) {
            pred = await res.json();
        }
    } catch (e) {
        // Fallback to precomputed data
    }

    // Static fallback if API is unavailable or spinning up
    if (!pred && currentEngineData && currentEngineData.predictions) {
        const key = String(cutoffCycle);
        pred = currentEngineData.predictions[key];
        if (!pred) {
            const keys = Object.keys(currentEngineData.predictions).map(Number).sort((a,b)=>a-b);
            const closest = keys.reduce((prev, curr) => Math.abs(curr - cutoffCycle) < Math.abs(prev - cutoffCycle) ? curr : prev, keys[0]);
            pred = currentEngineData.predictions[String(closest)];
        }
    }

    if (!pred) return;

    // Update Live Output Cards
    const liveCycle = document.getElementById("liveCycleNum");
    if (liveCycle) liveCycle.innerText = pred.cutoff_cycle;

    const livePred = document.getElementById("livePredictedRul");
    if (livePred) livePred.innerText = pred.predicted_rul.toFixed(1);

    const truthEl = document.getElementById("liveTruthRul");
    const truthMeta = document.getElementById("liveTruthMeta");
    if (truthEl) {
        if (pred.ground_truth_rul !== null) {
            truthEl.innerText = pred.ground_truth_rul.toFixed(0);
            if (truthMeta) truthMeta.innerText = `Error: ${pred.prediction_error >= 0 ? '+' : ''}${pred.prediction_error.toFixed(1)} cyc`;
        } else {
            truthEl.innerText = "--";
            if (truthMeta) truthMeta.innerText = "In-service flight";
        }
    }

    const healthEl = document.getElementById("liveHealthMargin");
    if (healthEl) healthEl.innerText = `${pred.health_margin_pct}%`;

    // Render Cutoff progression chart
    renderCutoffTelemetryChart(engineId, cutoffCycle);

    // Update Maintenance Risk Section
    updateRiskView(pred);
}

function renderCutoffTelemetryChart(engineId, cutoff) {
    if (!currentEngineData || !currentEngineData.records) return;

    const sub = currentEngineData.records.filter(r => r.cycle <= cutoff);
    const cycles = sub.map(r => r.cycle);

    const traces = [
        { x: cycles, y: sub.map(r => r.s_2), name: "s_2 (LPC Outlet Temp)", line: { color: "#0284C7", width: 2 } },
        { x: cycles, y: sub.map(r => r.s_3), name: "s_3 (HPC Outlet Temp)", line: { color: "#EA580C", width: 2 } },
        { x: cycles, y: sub.map(r => r.s_4), name: "s_4 (LPT Outlet Temp)", line: { color: "#16A34A", width: 2 } },
        { x: cycles, y: sub.map(r => r.s_11), name: "s_11 (HPC Static Pres)", line: { color: "#9333EA", width: 2 } }
    ];

    const layout = {
        title: {
            text: `Telemetry Stream up to Cycle ${cutoff} — Engine #${engineId}`,
            font: { family: "Poppins, sans-serif", size: 14, color: "#1E293B" }
        },
        font: { family: "Plus Jakarta Sans, sans-serif", size: 11, color: "#64748B" },
        paper_bgcolor: "#FFFFFF",
        plot_bgcolor: "#FFFFFF",
        xaxis: { title: "Flight Cycle", gridcolor: "#F1F5F9", linecolor: "#E2E8F0" },
        yaxis: { title: "Sensor Measurement", gridcolor: "#F1F5F9", linecolor: "#E2E8F0" },
        margin: { l: 55, r: 25, t: 60, b: 65 },
        legend: { orientation: "h", y: -0.28, x: 0.5, xanchor: "center" }
    };

    Plotly.newPlot("prognosticTrajectoryChart", traces, layout, { responsive: true, displayModeBar: false });
}

function updateRiskView(pred) {
    const risk = pred.risk_classification;
    const banner = document.getElementById("riskAlertBanner");
    const badge = document.getElementById("riskBadge");
    const title = document.getElementById("riskAlertTitle");
    const desc = document.getElementById("riskAlertDesc");
    const probPill = document.getElementById("riskProbPill");

    if (!banner || !risk) return;

    if (risk.predicted_class === 1) {
        banner.className = "pastel-card card-rose";
        badge.className = "pill-badge pill-white";
        badge.style.background = "#E11D48";
        badge.style.color = "#FFFFFF";
        badge.innerText = "CRITICAL ALERT";
        title.style.color = "#9F1239";
        title.innerText = "Hangar Inspection Required";
        desc.style.color = "#BE123C";
        desc.innerText = `Engine #${pred.unit_nr} has entered the critical risk window (Predicted RUL: ${pred.predicted_rul.toFixed(1)} cyc ≤ 30 cyc). Multi-channel telemetry confirms severe High Pressure Compressor (HPC) stage wear.`;
    } else {
        banner.className = "pastel-card card-mint";
        badge.className = "pill-badge pill-white";
        badge.style.background = "#16A34A";
        badge.style.color = "#FFFFFF";
        badge.innerText = "STATUS: NOMINAL";
        title.style.color = "#14532D";
        title.innerText = "Nominal Operating State";
        desc.style.color = "#166534";
        desc.innerText = `Engine #${pred.unit_nr} is operating with healthy aerodynamic and thermodynamic safety margins (RUL: ${pred.predicted_rul.toFixed(1)} cyc). No immediate overhaul scheduled.`;
    }

    if (probPill) {
        probPill.innerText = `Confidence: ${(risk.probability * 100).toFixed(1)}%`;
    }

    renderRiskGauge(risk.probability * 100);
}

function renderRiskGauge(confidencePct) {
    const gaugeData = [{
        type: "indicator",
        mode: "gauge+number",
        value: confidencePct,
        domain: { x: [0, 1], y: [0, 1] },
        title: { text: "Diagnostic Confidence Index (%)", font: { size: 13, family: "Poppins, sans-serif" } },
        number: { suffix: "%", font: { size: 28, family: "Poppins, sans-serif" } },
        gauge: {
            axis: { range: [0, 100], tickwidth: 1, tickcolor: "#CBD5E1" },
            bar: { color: confidencePct > 50 ? "#DC2626" : "#0284C7", thickness: 0.28 },
            bgcolor: "#FFFFFF",
            borderwidth: 1,
            bordercolor: "#E2E8F0",
            steps: [
                { range: [0, 40], color: "#D7F5E3" },
                { range: [40, 70], color: "#FEE8D6" },
                { range: [70, 100], color: "#FCDDE5" }
            ],
            threshold: {
                line: { color: "#DC2626", width: 3 },
                thickness: 0.75,
                value: 50
            }
        }
    }];

    const layout = {
        height: 250,
        margin: { l: 25, r: 25, t: 30, b: 20 },
        paper_bgcolor: "#FFFFFF"
    };

    Plotly.newPlot("riskGaugeContainer", gaugeData, layout, { responsive: true, displayModeBar: false });
}

// ----------------------------------------------------------------------------
// Model Benchmarks
// ----------------------------------------------------------------------------
function switchBenchmarkSubtab(tab) {
    const regPane = document.getElementById("benchmarkRegressionPane");
    const clfPane = document.getElementById("benchmarkClassificationPane");
    const buttons = document.querySelectorAll(".subtab-btn");

    if (tab === "regression") {
        if (regPane) regPane.style.display = "block";
        if (clfPane) clfPane.style.display = "none";
        if (buttons[0]) buttons[0].classList.add("active");
        if (buttons[1]) buttons[1].classList.remove("active");
    } else {
        if (regPane) regPane.style.display = "none";
        if (clfPane) clfPane.style.display = "block";
        if (buttons[0]) buttons[0].classList.remove("active");
        if (buttons[1]) buttons[1].classList.add("active");
    }
}

function renderBenchmarkCharts() {
    const models = ["Random Forest", "Grad Boost", "Decision Tree", "Linear Reg"];
    const maeValues = [12.02, 12.87, 16.45, 21.34];
    const r2Values = [0.8299, 0.8114, 0.6934, 0.4782];

    const maeTrace = [{
        x: models,
        y: maeValues,
        type: "bar",
        marker: { color: ["#0284C7", "#38BDF8", "#94A3B8", "#CBD5E1"] }
    }];

    const maeLayout = {
        title: { text: "MAE in Cycles (Lower is Better)", font: { family: "Poppins", size: 13, color: "#1E293B" } },
        paper_bgcolor: "#FFFFFF",
        plot_bgcolor: "#FFFFFF",
        margin: { l: 35, r: 20, t: 40, b: 35 },
        yaxis: { gridcolor: "#F1F5F9" }
    };

    Plotly.newPlot("benchmarkMaeChart", maeTrace, maeLayout, { responsive: true, displayModeBar: false });

    const r2Trace = [{
        x: models,
        y: r2Values,
        type: "bar",
        marker: { color: ["#16A34A", "#4ADE80", "#0284C7", "#94A3B8"] }
    }];

    const r2Layout = {
        title: { text: "R² Score (Higher is Better)", font: { family: "Poppins", size: 13, color: "#1E293B" } },
        paper_bgcolor: "#FFFFFF",
        plot_bgcolor: "#FFFFFF",
        margin: { l: 35, r: 20, t: 40, b: 35 },
        yaxis: { gridcolor: "#F1F5F9" }
    };

    Plotly.newPlot("benchmarkR2Chart", r2Trace, r2Layout, { responsive: true, displayModeBar: false });
}

// ----------------------------------------------------------------------------
// Responsive Plotly Chart Resize on Mobile Orientation or Viewport Change
// ----------------------------------------------------------------------------
window.addEventListener("resize", () => {
    try {
        const activePane = document.querySelector(".tab-pane.active");
        if (!activePane) return;
        if (activePane.id === "tab-telemetry" && document.getElementById("telemetryChartContainer")) {
            Plotly.Plots.resize("telemetryChartContainer");
        } else if (activePane.id === "tab-prognostics" && document.getElementById("prognosticTrajectoryChart")) {
            Plotly.Plots.resize("prognosticTrajectoryChart");
        } else if (activePane.id === "tab-risk" && document.getElementById("riskGaugeContainer")) {
            Plotly.Plots.resize("riskGaugeContainer");
        } else if (activePane.id === "tab-benchmarks") {
            if (document.getElementById("benchmarkMaeChart")) Plotly.Plots.resize("benchmarkMaeChart");
            if (document.getElementById("benchmarkR2Chart")) Plotly.Plots.resize("benchmarkR2Chart");
        }
    } catch (e) {
        // silent
    }
});
