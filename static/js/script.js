/**
 * SUBATHRA D - E-Commerce Customer Satisfaction Prediction
 * Frontend Interactive Logic & Helpers
 */

document.addEventListener("DOMContentLoaded", function () {
  // -------------------------------------------------------------
  // 1. Quick-Fill Sample Profiles (For instant live demo)
  // -------------------------------------------------------------
  const sampleProfiles = {
    high: {
      customer_age: 32,
      gender: "Female",
      product_category: "Electronics",
      purchase_amount: 28500.0,
      delivery_time: 2,
      service_interaction: 5,
      previous_orders: 14,
      discount_used: "Yes",
      payment_method: "UPI",
      website_rating: 5,
      return_requested: "No",
      order_frequency: "Weekly",
    },
    medium: {
      customer_age: 44,
      gender: "Male",
      product_category: "Fashion",
      purchase_amount: 4500.0,
      delivery_time: 5,
      service_interaction: 3,
      previous_orders: 4,
      discount_used: "Yes",
      payment_method: "Credit Card",
      website_rating: 3,
      return_requested: "No",
      order_frequency: "Monthly",
    },
    low: {
      customer_age: 59,
      gender: "Male",
      product_category: "Home & Kitchen",
      purchase_amount: 19800.0,
      delivery_time: 10,
      service_interaction: 1,
      previous_orders: 0,
      discount_used: "No",
      payment_method: "Cash on Delivery",
      website_rating: 1,
      return_requested: "Yes",
      order_frequency: "Rarely",
    },
  };

  window.fillSampleData = function (type) {
    const profile = sampleProfiles[type];
    if (!profile) return;

    for (const [key, value] of Object.entries(profile)) {
      const field = document.getElementById(key);
      if (field) {
        field.value = value;
        // Trigger subtle animation
        field.style.transition = "background-color 0.4s ease";
        field.style.backgroundColor = "#eef2ff";
        setTimeout(() => {
          field.style.backgroundColor = "#ffffff";
        }, 500);
      }
    }
  };

  // -------------------------------------------------------------
  // 2. Tab Navigation Logic (Dashboard)
  // -------------------------------------------------------------
  const tabButtons = document.querySelectorAll(".tab-btn");
  if (tabButtons.length > 0) {
    tabButtons.forEach((btn) => {
      btn.addEventListener("click", () => {
        const targetTab = btn.getAttribute("data-tab");

        // Remove active class from buttons & panes
        tabButtons.forEach((b) => b.classList.remove("active"));
        document.querySelectorAll(".tab-pane").forEach((pane) => {
          pane.classList.remove("active");
        });

        // Activate selected
        btn.classList.add("active");
        const activePane = document.getElementById(targetTab);
        if (activePane) {
          activePane.classList.add("active");
        }
      });
    });
  }

  // -------------------------------------------------------------
  // 3. Prediction History Search & Filter
  // -------------------------------------------------------------
  const historySearch = document.getElementById("historySearch");
  const filterSatisfaction = document.getElementById("filterSatisfaction");
  const historyTableBody = document.getElementById("historyTableBody");

  function filterHistoryRows() {
    if (!historyTableBody) return;
    const query = (historySearch ? historySearch.value : "").toLowerCase().trim();
    const filter = (filterSatisfaction ? filterSatisfaction.value : "ALL");

    const rows = historyTableBody.querySelectorAll("tr");
    let visibleCount = 0;

    rows.forEach((row) => {
      const text = row.innerText.toLowerCase();
      const sat = row.getAttribute("data-satisfaction") || "";

      const matchesQuery = query === "" || text.includes(query);
      const matchesFilter = filter === "ALL" || sat.toUpperCase() === filter.toUpperCase();

      if (matchesQuery && matchesFilter) {
        row.style.display = "";
        visibleCount++;
      } else {
        row.style.display = "none";
      }
    });

    const countBadge = document.getElementById("recordsCount");
    if (countBadge) {
      countBadge.innerText = `${visibleCount} records showing`;
    }
  }

  if (historySearch) {
    historySearch.addEventListener("input", filterHistoryRows);
  }
  if (filterSatisfaction) {
    filterSatisfaction.addEventListener("change", filterHistoryRows);
  }
});
