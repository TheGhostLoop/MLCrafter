import os,numpy as np,pandas as pd,matplotlib.pyplot as plt,numpy as np,joblib,json
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

def create_df_structure():
    # first creating columns
    name =  input("Enter name for your DataFrame:: ")
    n_of_columns = int(input("Enter number of columns: "))
    colsname = []
    for i in range(n_of_columns):
        name = input(f"Enter name for column {i+1}: ")
        colsname.append(name)
    df = pd.DataFrame(columns=colsname)

    # taking rows input
    while True:
        cname = ",".join(df.columns)
        choice = (input('''
1 - Insert data into dataframe
2 - Exit
Choice: '''))


        if choice == "1":
            row = input(f'''
Feed data like this: 1,Alice,90
Separate values by comma (NO brackets)
{cname}: ''')
            data = row.strip().split(',')
            if len(data) != len(colsname):
                print("[->]Incorrect number of values. Try again.\n")
                continue

            df.loc[len(df)] = data
        elif choice == "2":
            break
        else:
            print("[->]Invalid choice. Try again.")

    # Save to CSV
    os.makedirs('dataframes' ,exist_ok=True)
    filepath = f"dataframes/{name}.csv"
    df.to_csv(f'{filepath}', index=False)
    print(f"\n✅ Data saved to '{name}.csv'")
    print(df)

def isfloat(num):
    try:
        val = float(num)
        return True
    except ValueError:
        return False

def select_df():
    try:
        print("Available files -")
        for i in os.listdir("dataframes"):
            print("[->]",i)
        filename = input('Please enter filename:: ')
        filepath = f'dataframes/{filename}'
        df = pd.read_csv(filepath) # reading for verifying only
        print(f'[->] {filename} ✅ has been selected!!')
        print(f'Data looks like ->\n{df.head()}\n')
        train_model(filepath)
    except FileNotFoundError:
        print(f'File {filename} ❌ Not Found!\nRedirecting again...\n\n')




def makeprediction(l_inames,dummiescols,model,o_name):
        try:
            pdatainput = input(f'''
    Enter data! 👇
    Format: {l_inames} (separated by ,)
    [->]: ''')
            pdatainput_final = pdatainput.strip().split(',')
            a = []
            for i in pdatainput_final:
                if(i.isnumeric() == True or isfloat(i) == True):
                    a.append(float(i))
                else:
                    a.append(i)
            finaldata = pd.DataFrame([a],columns=l_inames)
            finaldata = pd.get_dummies(finaldata,drop_first=True)
            for i in dummiescols:
                if i not in finaldata.columns:
                    finaldata[i] = 0
            finaldata = finaldata[dummiescols]


            # printing prediction result!
            result = model.predict(finaldata)
            print(f'Predicted result:: -> {result[0]:.2f} {o_name}')

        except:
            print(f"Input Data Not Matched!!\nRedirecting to Previous Menu\n")

            return

def plotgraph(l_inames,input_data,output_data,model,o_name):
        g_column = input(f"Enter any one column name\nAvailable Choices:{l_inames}\n[->]: ")
        try:
            test = input_data[g_column]
        except:
            print(f"Graph is not possible on {g_column}\nPlease choose different column.")
            g_column = input(f"Enter any one column name\nAvailable Choices:{l_inames}\n[->]: ")
        plt.plot(input_data[g_column],output_data,label=f'Actual data',color='orange',marker='o')
        plt.plot(input_data[g_column],model.predict(input_data),label=f'Predicted data',color='green',marker='o')
        plt.xlabel(f'{g_column}')
        plt.ylabel(f'{o_name}')
        plt.grid(True)
        plt.legend()
        plt.title('Linear Regression Model!')
        plt.show()


def train_model(filepath):
    df = pd.read_csv(filepath) 
    allcolumns = ",".join(df.columns)  
    i_names = input(f'''
    Kindly enter input columns names!
                    eg: age,experience,salary
                    then choose (age,experience)
                Available columns:: {allcolumns}
                [->]: ''')
    o_name = input(f'''
    Enter prediction column!
                    eg: salary
                    Choose only one:: {allcolumns}
                [->]: ''')
    


    # dividing data into inputs and outputs here 
    l_inames = i_names.strip().split(",")
    input_data = df[l_inames]
    output_data = df[o_name]


    # handling string columns
    input_data = pd.get_dummies(input_data,drop_first=True)
    dummiescols = input_data.columns.tolist()




    # starting training here
    t_value = input('''Enter test size!\neg (0.2 = Train with (80%) & test with (20%) data )\nd -> For default\nRange: 0 to 1\n[->]: ''')
    if(t_value.lower()=='d'):
        t_size = 0.2
    else:
        t_size = float(t_value)
    
    input_train,input_test,output_train,output_test = train_test_split(input_data,output_data,test_size=t_size,random_state=42)

    model = LinearRegression()
    model.fit(input_train,output_train)

    predicted_data = model.predict(input_test)

    r2 = r2_score(predicted_data,output_test)
    print(f'''
Model Trained! ✅
        [->] Your r2_Score = {r2:.2f}
        [->] Your Root_mean_Squared = {np.sqrt(mean_squared_error(output_test,predicted_data)):.2f}
        [->] Your Mean_Absolute_Error = {mean_absolute_error(output_test,predicted_data):.2f}
''')
    
    #  saving trained model!!
    name = input('Enter a name for your model\n[->]: ')
    os.makedirs('trained_models',exist_ok=True)

    bundle = {
    "model": model,
    "l_inames": l_inames,
    "dummiescols": dummiescols,
    "o_name": o_name,
    "input_data": input_data,
    "output_data": output_data
    }


    joblib.dump(bundle, f"trained_models/{name}_bundle.pkl")
    # joblib.dump(model,f"trained_models/{name}.pkl")

    # joblib.dump(dummiescols,colpath)

    print(f"{name} Saved!! ✅")
    return


while(1):
    choice = (input("1 - Create DF for ML Model.\n2 - Train Your Own ML Model\n3 - Use Existing Models\n4 - Exit\n[->]: "))
    if(choice=="1"):
        create_df_structure()
    elif(choice=="2"):
        select_df()
    elif(choice=="3"):

        print("\n\nAvailable Trained Models 👇")

        for i in os.listdir("trained_models"):
            print(i)

        modelname = input("Enter fullname [->]: ")


        try:
            bundle = joblib.load(f"trained_models/{modelname}")
            model = bundle["model"]
            l_inames = bundle["l_inames"]
            dummiescols = bundle["dummiescols"]
            input_data = bundle["input_data"]
            output_data = bundle["output_data"]
            o_name = bundle["o_name"]
            print(f"{modelname} Loaded ✅")
        except:
            print("❌ Model not found.")
            continue
        while(1):
            c = (input("1 -> Show Graph\n2 -> Make Prediction\n3 -> Main Menu\n[->]: "))
            if(c=="1"):
                plotgraph(l_inames,input_data,output_data,model,o_name)
            elif(c=="2"):
                makeprediction(l_inames,dummiescols,model,o_name)
            elif(c=="3"):
                break
            else:
                print('Wrong Choice')


    elif(choice==4):
        print("Bye!\nExiting Program!")
        break
    else:
        print("Wrong Choice!!")
