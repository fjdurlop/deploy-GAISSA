# Steps to Get an AWS Free Tier Virtual Machine

1. Get a free AWS account:
- https://aws.amazon.com/free/?sc_icampaign=acq_freetier-default&sc_ichannel=ha&sc_icontent=awssm-evergreen-default&sc_iplace=ed&trk=ha_awssm-evergreen-default&all-free-tier.sort-by=item.additionalFields.SortRank&all-free-tier.sort-order=asc&awsf.Free%20Tier%20Types=*all&awsf.Free%20Tier%20Categories=*all

<center><figure>
  <img
  src="images/aws_free_tier_01.png"
</figure></center>
<p style="text-align: center;">AWS free tier account creation.</p>

1. **Sign in to AWS Console:** Go to the AWS Management Console (https://console.aws.amazon.com/) and sign in with your AWS account credentials. Sign up if you do not have an account.

2. setup budget
https://aws.amazon.com/getting-started/hands-on/control-your-costs-free-tier-budgets/

2. **Navigate to EC2 Service:** Once logged in, navigate to the EC2 service by searching for "EC2" in the search bar at the top of the console.

3. **Launch Instance:** In the EC2 Dashboard, click on the "Launch Instance" button to start the process of launching a new virtual machine.

1. **Select an Amazon Machine Image (AMI):** Choose an Amazon Machine Image (AMI) from the available options. For a free tier eligible instance, you can choose an image labeled "Free tier eligible" in the AMI selection page.
- select ubuntu
<center><figure>
  <img
  src="images/aws_instance_01.png"
</figure></center>
<p style="text-align: center;">VM image.</p>

2. **Choose an Instance Type:** Select the desired instance type that falls under the free tier eligible category. You can review the details and specifications of each instance type to choose the one that suits your requirements.

3. **Configure Instance:** Configure the instance details, such as the number of instances to launch, network settings, and storage options. You can leave most of the settings as default for now, but make sure to select the appropriate VPC, subnet, and security group settings.

4. **Add Storage:** Specify the storage options for your instance. The default storage size is usually sufficient for basic usage, but you can adjust it according to your needs. We use 30GB in this case


6.  **Configure Security Group:** Configure the security group settings to control inbound and outbound traffic to your instance. Make sure to allow SSH access by adding a rule that allows inbound SSH (port 22) connections from your IP address or a specific IP range.

7.  **Review and Launch:** Review all the settings you have configured for your instance. Double-check if everything looks correct, and then click on the "Launch" button to proceed.
8.  **SSH conection:**
9.  **Create a Key Pair:** In the key pair selection page, choose to either create a new key pair or use an existing one. If creating a new key pair, follow the instructions to download the private key file (.pem). This key pair is required to connect to your instance via SSH.

10. **Launch Instances:** Click on the "Launch Instances" button to launch your virtual machine(s) based on the chosen configuration.

11. **Access your Instance via SSH:** Once the instances are launched, you can connect to them using SSH. Open your terminal or SSH client and use the downloaded private key file to establish an SSH connection to your instance. The command typically looks like this:

   ```shell
   ssh -i /path/to/private_key.pem username@public_dns_name
   ```
Replace /path/to/private_key.pem with the actual path to your private key file, username with the appropriate username (depending on the chosen AMI), and public_dns_name with the public DNS name or IP address of your instance.

    Start Using Your Free Tier Virtual Machine: You are now connected to your AWS free tier virtual machine. You can start using it for your desired purposes, such as running applications, deploying websites, or performing data analysis.