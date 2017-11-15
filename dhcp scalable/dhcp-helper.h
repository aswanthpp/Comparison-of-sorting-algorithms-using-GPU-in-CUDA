#ifndef DHCP_HELPER_H
#define DHCP_HELPER_H

#include <stdint.h>
#include "ns3/application-container.h"
#include "ns3/net-device-container.h"
#include "ns3/object-factory.h"
#include "ns3/ipv4-address.h"
#include "ns3/ipv4-interface-container.h"

namespace ns3 {

    /*The helper class used to configure and install DHCP applications on nodes*/
	class DhcpHelper
	{
	public:
		DhcpHelper ();

         /*Set DHCP client attributes*/
		void SetClientAttribute (std::string name, const AttributeValue &value);

        /*Set DHCP server attributes*/
		void SetServerAttribute (std::string name, const AttributeValue &value);

        void SetRelayAttribute (std::string name,const AttributeValue &value);

        /*Install DHCP client of a node / NetDevice
        netDevice : The NetDevice that the DHCP client will use
        returns : The application container with DHCP client installed*/
		ApplicationContainer InstallDhcpClient (Ptr<NetDevice> netDevice) const;

        /*Install DHCP client of a set of nodes / NetDevices
        netDevices : The NetDevices that the DHCP client will use
        returns : The application container with DHCP client installed*/
		ApplicationContainer InstallDhcpClient (NetDeviceContainer netDevices) const;


        ApplicationContainer InstallDhcpRelay (NetDeviceContainer netDevices,Ipv4Address relayAddress,Ipv4Address relayAddressClientSide,
                                                            Ipv4Address dhcps,Ipv4Mask poolMask) const;


        /*Install DHCP server of a node / NetDevice
        Note: the server address must be coherent with the pool address, because
        DHCP relays are not yet supported.
        netDevice : The NetDevice on which DHCP server application has to be installed
        serverAddr : The Ipv4Address of the server
        poolAddr : The Ipv4Address (network part) of the allocated pool
        poolMask : The mask of the allocated pool
        minAddr : The lower bound of the Ipv4Address pool
        maxAddr : The upper bound of the Ipv4Address pool
        gateway : The Ipv4Address of default gateway (optional)
        returns : The application container with DHCP server installed*/
		ApplicationContainer InstallDhcpServer (Ptr<NetDevice> netDevice, Ipv4Address serverAddr,
			Ipv4Address poolAddr, Ipv4Mask poolMask,
			Ipv4Address minAddr, Ipv4Address maxAddr,
			Ipv4Address gateway = Ipv4Address ());


    /*    ApplicationContainer InstallDhcpRelay (Ptr<NetDevice> netDevice, Ipv4Address relayAddr,
                                                     Ipv4Mask subMask, Ipv4Address dhcps);*/
        /*Assign a fixed IP addresses to a net device.
        netDevice : The NetDevice on which the address has to be installed
        addr : The Ipv4Address
        mask : The network mask
        returns : the Ipv4 interface container*/
		Ipv4InterfaceContainer InstallFixedAddress (Ptr<NetDevice> netDevice, Ipv4Address addr, Ipv4Mask mask);

	private:
        /*Function to install DHCP client on a node
        netDevice : The NetDevice on which DHCP client application has to be installed
        returns : The pointer to the installed DHCP client*/
		Ptr<Application> InstallDhcpClientPriv (Ptr<NetDevice> netDevice) const;

       Ptr<Application> InstallDhcpRelayClientSide(Ptr<NetDevice> netDevice,Ipv4Address relayAddressClientSide,Ipv4Mask mask) const;
       Ptr<Application> InstallDhcpRelayServerSide(Ptr<NetDevice> netDevice,Ipv4Address relayAddress,Ipv4Address dhcps,Ipv4Mask mask) const;

        /*ObjectFactory : Instantiate subclasses of ns3::Object. This class can also hold a set of attributes to 
        set automatically during the object construction*/
		ObjectFactory m_clientFactory;                 
		ObjectFactory m_serverFactory;
        ObjectFactory m_relayFactory;                 

        /*list of fixed addresses already allocated*/
		std::list<Ipv4Address> m_fixedAddresses;  

        /*list of address pools*/     
		std::list< std::pair <Ipv4Address, Ipv4Address> > m_addressPools; 
	};

}

#endif 
