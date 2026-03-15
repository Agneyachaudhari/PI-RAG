# PI-RAG
Makes it very easy to run you own, local, RAG system on a Raspberry Pi 5 with 4gb of ram or above.

#Hardware & software needed
-Raspberry Pi 5
-Raspberry pi os lite(64 bit)
-Complementary components(PSU, sd card, etc)
-An intial internet connection for the setup(can be removed after setup)

#Setup
In the File where you want the program run
```bash
git clone https://github.com/Agneyachaudhari/PI-RAG
```
In the terminal.

After it runs you wnat to run
```bash
cd PI-RAG
```
Then finally run
```bash
bash setup.sh
```
This will take a while.

After it is done it should show a success message.
 Run 
```bash
rm setup.sh
```
You have now succesfully installed your RAG system.
To put data in the rag system make sure it is a .txt file and copy it to the **data** directory
