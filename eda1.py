import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

data= pd.read_csv('9. Sales-Data-Analysis.csv')

def smart_understand(df):
    st.header("🔎 Dataset Summary")

    # Shape & Columns
    st.subheader("Basic Info")
    st.write("Shape:", df.shape)
    st.write("Columns:", list(df.columns))

    # Data Types & Missing
    st.subheader("Data Types & Missing Values")
    info = pd.DataFrame({
        "Data Type": df.dtypes,
        "Missing Values": df.isnull().sum(),
        "Unique Values": df.nunique(),
        "First Value": df.iloc[0],
        "Last Value": df.iloc[-1]
    })
    st.write(info)

    # Unique identity of each column
    st.subheader("Unique Values...")
    col= st.selectbox("Choose a column:", ["Order ID", "Date" ,"Product","Price","Quantity","Purchase Type","Payment Method","Manager","City","Sales"])

    st.write(df[col].unique().T)
    # Quick Stats for numeric
    st.subheader("Descriptive Statistics (Numeric)")
    st.write(df.describe().T)

    # Quick Stats for categorical
    st.subheader("Descriptive Statistics (Categorical)")
    st.write(df.describe(include="object").T)

    # Correlation (just values, no heatmap)
    st.subheader("Correlation (Numeric Only)")
    st.write(df.corr(numeric_only=True))

    # Sample Records
    st.subheader("Sample Records")
    rows = st.slider("Select number of rows to preview:", 5, 50, 5)
    st.write(df.sample(rows))

# smart_understand(data)

# Adding some important things for better visualization
def ad(df):
    # Create a new column 'Sales' (Quantity × Price)
    df["Sales"] = df["Quantity"].round(0) * df["Price"].round(0)

    # Save back to CSV (overwrite the same file)
    df.to_csv("9. Sales-Data-Analysis.csv", index=False)

    return df
    
data = ad(data)
# Data Cleaning
def data_clean(df):
    st.write("### Starting Data Cleaning...")

    # 2. Drop duplicates
    dup = df.duplicated().sum()   # number of duplicate rows
    if dup > 0:
        df = df.drop_duplicates()
        st.write(f"✅ Dropped {dup} duplicate rows")
    else:
        st.write("ℹ️ No duplicate rows found")

    # 3. Handle missing values
    nul = df.isnull().sum().sum()   # total number of null values
    if nul > 0:
        df = df.fillna(df.mean(numeric_only=True))   # only numeric columns
        st.write(f"✅ Filled {nul} missing values with column mean")
    else:
        st.write("ℹ️ No missing values found")

    # Fix Data Types
    def fix(df):
        df['Order ID'] = pd.to_numeric(df['Order ID'], errors='coerce')
        # df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

        # Product
        for col in df.select_dtypes(include='object'):      # String cleaning
            df[col] = df[col].str.strip()
            df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
            df = df.apply(lambda x: " ".join(x.split()) if isinstance(x, str) else x)
        st.write("✅ Extra spaces removed")


        for col in df.select_dtypes(include=["float"]).columns:
            df[col] = df[col].round(0).astype("Int64")  # round to whole numbers

        return df

    df = fix(df)

    st.write("✅ Data types fixed and numeric columns rounded")
    st.title("Cleaned data...")
    st.write(df)

    return df
    
# data_clean(data)

# Data Visualization
def visual(df):

    col = st.selectbox("Choose a column:", ["Price", "Manager"])

    if col == 'Price':
        fig, ax= plt.subplots()
        bx= sns.countplot(x=col, data= df, ax=ax)

        for bars in ax.containers:
            ax.bar_label(bars)
        st.pyplot(fig)

        st.markdown("<span style='color:beige; font-weight:normal; font-size:16px; '>Most items are sold at the lowest price (3) → very high volume.</span>", unsafe_allow_html=True)

        st.markdown("<span style='color:beige; font-weight:normal; font-size:16px;'>Mid-range prices (5, 10, 13) also sell decently.</span>", unsafe_allow_html=True)

        st.markdown("<span style='color:beige; font-weight:normal; font-size:16px; '>High prices (26, 29) sell extremely rarely.</span>", unsafe_allow_html=True)
        

    else:
        fig, ax= plt.subplots(figsize= (10,6))
        bx= sns.countplot(x=col, data= df, ax=ax)

        for bars in ax.containers:
            ax.bar_label(bars)
        st.pyplot(fig)
        st.markdown("<span style='color: light brown; font-weight: bold; font-size:16px; '>Performance is uneven. Most sales are concentrated with Tom Jackson and Joao Silva, while others contribute much less.</span>", unsafe_allow_html=True)
    col2= st.selectbox("Choose a column:", ["Quantity","None"])
    if col2== "Quantity":
        plt.figure(figsize=(8,6))
        sns.kdeplot(x=df["Quantity"], y=df["Sales"], levels=10, cmap="magma")
        plt.title("Quantity Analysis", fontsize=14, weight="bold")
        st.pyplot(plt)
        st.write("Most sales come from small orders, but a few big or premium purchases bring in high revenue.")
        st.write("Focus:Small orders → for steady revenue & Big orders → for high profit")
    else:
        st.subheader("Lets move furthur..")

    col3=st.selectbox("Choose a column:",["Purchase Type", "Payment Method","Manager","City", "Product"],key="col3")
    if col3== "Purchase Type":
        co= st.selectbox("Choose a column:",["Product & City", "Sales"],key="purchase_type_option")
        if co == "Product & City":
            g = sns.FacetGrid(df, col="Product")
            g.map_dataframe(sns.histplot, x="Purchase Type", hue="City",  multiple="dodge",shrink=0.8)
            g.add_legend()
            st.markdown("<span style='color: Blue; font-weight: bold; font-size:16px; '>Blue → London</span>", unsafe_allow_html=True)
            st.markdown("<span style='color: Orange; font-weight: bold; font-size:16px; '> Orange → Paris</span>", unsafe_allow_html=True)
            st.markdown("<span style='color: Light Green; font-weight: bold; font-size:16px; '> Green → Berlin  </span>", unsafe_allow_html=True)
            st.markdown("<span style='color: Red; font-weight: bold; font-size:16px; '> Red → Madrid  </span>", unsafe_allow_html=True)
            st.markdown("<span style='color: Purple; font-weight: bold; font-size:16px; '>  Purple → Rome </span>", unsafe_allow_html=True)
            st.pyplot(g.fig)
            st.markdown("<span style='color: light brown; font-weight: bold; font-size:16px; '>This chart shows how different cities (each colour) buy different products across Online, In-store, and Drive-thru.</span>", unsafe_allow_html=True)

        else:
            total_sales = df.groupby("Purchase Type")["Sales"].sum().reset_index()
            fig, ax = plt.subplots(figsize=(10,6))
            sns.barplot(x="Purchase Type", y="Sales", data=total_sales, estimator=sum, ax=ax)
            ax.set_title("Total Revenue per Product")
            st.pyplot(fig)
            st.markdown("<span style='color: light brown; font-weight: bold; font-size:16px; '>Most of the Sales done through online payment. </span>", unsafe_allow_html=True)
    elif col3 == "Payment Method":
        co = st.selectbox("Choose a column:", ["City", "Product"])

        if co == "City":
            cities = st.multiselect("Select 2 Cities:", df["City"].unique(), default=df["City"].unique()[:2])
            if len(cities) == 2:
                labels = df["Payment Method"].unique()
                left_vals = df[df["City"] == cities[0]]["Payment Method"].value_counts().reindex(labels, fill_value=0)
                right_vals = df[df["City"] == cities[1]]["Payment Method"].value_counts().reindex(labels, fill_value=0)

                fig, ax = plt.subplots()
                ax.barh(labels, left_vals, color="skyblue", label=cities[0])
                ax.barh(labels, -right_vals, color="orange", label=cities[1])
                ax.legend()
                ax.set_title(f"Payment Method: {cities[0]} vs {cities[1]}")
                st.pyplot(fig)
            else:
                st.warning("Please select exactly 2 cities.")

        elif co == "Product":
            products = st.multiselect("Select 2 Products:", df["Product"].unique(), default=df["Product"].unique()[:2])
            if len(products) == 2:
                labels = df["Payment Method"].unique()
                left_vals = df[df["Product"] == products[0]]["Payment Method"].value_counts().reindex(labels, fill_value=0)
                right_vals = df[df["Product"] == products[1]]["Payment Method"].value_counts().reindex(labels, fill_value=0)

                fig, ax = plt.subplots()
                ax.barh(labels, left_vals, color="lightgreen", label=products[0])
                ax.barh(labels, -right_vals, color="salmon", label=products[1])
                ax.legend()
                ax.set_title(f"Payment Method: {products[0]} vs {products[1]}")
                st.pyplot(fig)
            else:
                st.warning("Please select exactly 2 products.")
    elif col3 == "Manager":
        co = st.selectbox("Choose a column:", ["City", "Sales", "Product"])

    # Radar Chart helper function
        def radar_chart(categories, values_dict, title):
            N = len(categories)
            angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
            angles += angles[:1]  # close loop

            fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))

            for label, values in values_dict.items():
                vals = values + values[:1]  # close loop
                ax.plot(angles, vals, "o-", linewidth=2, label=label)
                ax.fill(angles, vals, alpha=0.25)

            ax.set_thetagrids(np.degrees(angles[:-1]), categories)
            plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
            plt.title(title)
            st.pyplot(fig)

        if co == "Sales":
            st.write("You selected: Manager vs Sales (Radar Chart)")

            categories = df["Manager"].unique().tolist()
            values_dict = {"Sales": df.groupby("Manager")["Sales"].sum().tolist()}

            radar_chart(categories, values_dict, "Manager vs Sales")

        elif co == "City":
            st.write("You selected: Manager vs City (Radar Chart)")

            categories = df["City"].unique().tolist()
            values_dict = {
                manager: df[df["Manager"] == manager]["City"].value_counts().reindex(categories, fill_value=0).tolist()
                for manager in df["Manager"].unique()
            }
            radar_chart(categories, values_dict, "Manager vs City")

        elif co == "Product":
            st.write("You selected: Manager vs Product (Radar Chart)")

            categories = df["Product"].unique().tolist()
            values_dict = {
            manager: df[df["Manager"] == manager]["Product"].value_counts().reindex(categories, fill_value=0).tolist()
            for manager in df["Manager"].unique()
        }
        radar_chart(categories, values_dict, "Manager vs Product")
    elif col3 == "City":
        co = st.selectbox("Choose a column:", ["Sales + Product", "Manager"])

        if co == "Sales + Product":
            st.write("You selected: City vs (Sales + Product) - Circular Bar Chart")

        # Example: group total sales by product
            grouped = df.groupby("Product")["Sales"].sum().reset_index()

            categories = grouped["Product"].tolist()
            values = grouped["Sales"].tolist()

            N = len(categories)
            theta = np.linspace(0.0, 2*np.pi, N, endpoint=False)
            radii = values
            width = 2*np.pi / N

            fig, ax = plt.subplots(subplot_kw={'projection':'polar'}, figsize=(8,8))
            bars = ax.bar(theta, radii, width=width, bottom=0)

            for bar in bars:
                bar.set_facecolor(plt.cm.viridis(np.random.rand()))
                bar.set_alpha(0.8)

            ax.set_xticks(theta)
            ax.set_xticklabels(categories, fontsize=10)
            plt.title("Circular Bar Chart (Product vs Total Sales)")
            st.pyplot(fig)

        elif co == "Manager":
            st.write("You selected: City vs Manager (Circular Bar Chart)")

        # Group total sales by Manager
            grouped = df.groupby(["Manager", "City"])["Sales"].sum().reset_index()

        # Create category labels as Manager|City
            categories = grouped["Manager"] + " | " + grouped["City"]
            values = grouped["Sales"].tolist()
            N = len(categories)
            theta = np.linspace(0.0, 2*np.pi, N, endpoint=False)
            radii = values
            width = 2*np.pi / N

            fig, ax = plt.subplots(subplot_kw={'projection':'polar'}, figsize=(8,8))
            bars = ax.bar(theta, radii, width=width, bottom=0)

            for bar in bars:
                bar.set_facecolor(plt.cm.viridis(np.random.rand()))
                bar.set_alpha(0.8)

            ax.set_xticks(theta)
            ax.set_xticklabels(categories, fontsize=9, rotation=45)
            plt.title("Circular Bar Chart: Manager vs Total Sales")
            st.pyplot(fig)
    elif col3 == "Product":
        co = st.selectbox("Choose a column:", ["Price", "Quantity", "Sales", "Purchase Type","Payment Method"])
    # Donut Charts
        if co in ["Price", "Purchase Type", "Payment Method"]:
            st.write(f"You selected: Product vs {co} (Donut Chart)")

            grouped = df.groupby([co, "Product"]).size().reset_index(name="Count")

            for product in df["Product"].unique():
                city_data = grouped[grouped["Product"] == product]

                sizes = city_data["Count"].tolist()
                labels = city_data[co].tolist()

                fig, ax = plt.subplots()
                wedges, texts, autotexts = ax.pie(
                    sizes, labels=labels, autopct="%1.1f%%", startangle=90
                )
                centre_circle = plt.Circle((0, 0), 0.6, fc="white")
                fig.gca().add_artist(centre_circle)
                ax.set_title(f"Donut Chart: {co} Distribution in {product}")
                st.pyplot(fig)

        elif co == "Quantity":
            st.write("You selected: City vs Quantity (Hexbin Chart)")

            # Convert City into numbers
            fig, ax = plt.subplots(figsize=(8,6))

            x = df["Product"].astype("category").cat.codes + np.random.uniform(-0.3, 0.3, size=len(df))  
            y = df["Quantity"]

            hb = ax.hexbin(x, y, gridsize=30, cmap="viridis")
            cb = fig.colorbar(hb, ax=ax)
            cb.set_label("Count")

            ax.set_xlabel("Product")
            ax.set_xticks(range(len(df["Product"].unique())))
            ax.set_xticklabels(df["Product"].unique(), rotation=45)
            ax.set_ylabel("Quantity")
            ax.set_title("Product vs Quantity (Hexbin)")

            st.pyplot(fig)
        elif co == "Sales":
            fig, ax = plt.subplots(figsize=(8,6))

            x = df["Product"].astype("category").cat.codes + np.random.uniform(-0.3, 0.3, size=len(df))
            y = df["Sales"]   # Sales is already numeric

            hb = ax.hexbin(x, y, gridsize=30, cmap="viridis")
            cb = fig.colorbar(hb, ax=ax)
            cb.set_label("Count")

            ax.set_xlabel("Product")
            ax.set_xticks(range(len(df["Manager"].unique())))
            ax.set_xticklabels(df["Product"].unique(), rotation=45)
            ax.set_ylabel("Sales")
            ax.set_title("Product vs Sales (Hexbin)")

            st.pyplot(fig)
st.write("1 for Data Understaning..")
st.write("2 for Cleaned Data...")
st.write("3 for Data Visualization....")
inp= st.text_input("Enter Your choice:")

if inp == "1":
    smart_understand(data)
elif inp == "2":
    data_clean(data)
elif inp== "3":
    visual(data_clean(data))