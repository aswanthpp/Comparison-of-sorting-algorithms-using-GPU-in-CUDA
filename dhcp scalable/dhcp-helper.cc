/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011 UPB
 * Copyright (c) 2017 NITK Surathkal
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Author: Radu Lupu <rlupu@elcom.pub.ro>
 *         Ankit Deepak <adadeepak8@gmail.com>
 *         Deepti Rajagopal <deeptir96@gmail.com>
 *
 */

#include "dhcp-helper.h"
#include "ns3/dhcp-server.h"
#include "ns3/dhcp-client.h"
#include "ns3/dhcp-relay.h"
#include "ns3/uinteger.h"
#include "ns3/names.h"
#include "ns3/ipv4.h"
#include "ns3/loopback-net-device.h"
#include "ns3/traffic-control-layer.h"
#include "ns3/traffic-control-helper.h"
#include "ns3/log.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("DhcpHelper");

DhcpHelper::DhcpHelper ()
{
  m_clientFactory.SetTypeId (DhcpClient::GetTypeId ());  // assigning dhcp_client type to m_clientFactory
  m_serverFactory.SetTypeId (DhcpServer::GetTypeId ()); // assigning dhcp_server type to m_serverFatory
  m_relayFactory.SetTypeId (DhcpRelay::GetTypeId ());   // assiging dhcp_relay type to m_relayFactory
}

void DhcpHelper::SetClientAttribute (
  std::string name,
  const AttributeValue &value)
{
  m_clientFactory.Set (name, value);      // adding attribute  to the client
}

void DhcpHelper::SetServerAttribute (
  std::string name,
  const AttributeValue &value)
{
  m_serverFactory.Set (name, value);
}

void DhcpHelper::SetRelayAttribute (
  std::string name,
  const AttributeValue &value)
{
  m_relayFactory.Set (name, value);
}

ApplicationContainer DhcpHelper::InstallDhcpClient (Ptr<NetDevice> netDevice) const
{
  return ApplicationContainer (InstallDhcpClientPriv (netDevice));
}

ApplicationContainer DhcpHelper::InstallDhcpClient (NetDeviceContainer netDevices) const
{
  ApplicationContainer apps;
  for (NetDeviceContainer::Iterator i = netDevices.Begin (); i != netDevices.End (); ++i) // inst
    {
      apps.Add (InstallDhcpClientPriv (*i));
    }
  return apps;
}

Ptr<Application> DhcpHelper::InstallDhcpClientPriv (Ptr<NetDevice> netDevice) const
{
  Ptr<Node> node = netDevice->GetNode ();
  NS_ASSERT_MSG (node != 0, "DhcpClientHelper: NetDevice is not not associated with any node -> fail");

  Ptr<Ipv4> ipv4 = node->GetObject<Ipv4> ();     // ipassociated with the netDevice
  NS_ASSERT_MSG (ipv4, "DhcpHelper: NetDevice is associated"
                 " with a node without IPv4 stack installed -> fail "
                 "(maybe need to use InternetStackHelper?)");

  int32_t interface = ipv4->GetInterfaceForDevice (netDevice);    ///  interface number of an ipv4
  if (interface == -1)
    {
      interface = ipv4->AddInterface (netDevice);    // if not add that interface
    }
  NS_ASSERT_MSG (interface >= 0, "DhcpHelper: Interface index not found");

  ipv4->SetMetric (interface, 1); /// cost of packet transmission through that link 1
  ipv4->SetUp (interface); 		 //// set the interface to be up


  // Install the default traffic control configuration if the traffic
  // control layer has been aggregated, if this is not
  // a loopback interface, and there is no queue disc installed already

  Ptr<TrafficControlLayer> tc = node->GetObject<TrafficControlLayer> ();
  if (tc && DynamicCast<LoopbackNetDevice> (netDevice) == 0 && tc->GetRootQueueDiscOnDevice (netDevice) == 0)
    {
      NS_LOG_LOGIC ("DhcpHelper - Installing default traffic control configuration");
      TrafficControlHelper tcHelper = TrafficControlHelper::Default ();  
      tcHelper.Install (netDevice);   
    }

  Ptr<DhcpClient> app = DynamicCast <DhcpClient> (m_clientFactory.Create<DhcpClient> ()); // create dhcpClient object
  app->SetDhcpClientNetDevice (netDevice);  // invoking a fnction of dhcpClient
  node->AddApplication (app);   // associate dhcpclient(arguement) to the node

  return app;
}

ApplicationContainer DhcpHelper::InstallDhcpServer (Ptr<NetDevice> netDevice, Ipv4Address serverAddr,
                                                    Ipv4Address poolAddr, Ipv4Mask poolMask,
                                                    Ipv4Address minAddr, Ipv4Address maxAddr,
                                                    Ipv4Address gateway)
{
  m_serverFactory.Set ("PoolAddresses", Ipv4AddressValue (poolAddr));  // assign the value to the variable PoolAddresses in Get TypeId
  m_serverFactory.Set ("PoolMask", Ipv4MaskValue (poolMask)); // similiarly above
  m_serverFactory.Set ("FirstAddress", Ipv4AddressValue (minAddr));
  m_serverFactory.Set ("LastAddress", Ipv4AddressValue (maxAddr));
  m_serverFactory.Set ("Gateway", Ipv4AddressValue (gateway));

  Ptr<Node> node = netDevice->GetNode ();
  NS_ASSERT_MSG (node != 0, "DhcpHelper: NetDevice is not not associated with any node -> fail");

  Ptr<Ipv4> ipv4 = node->GetObject<Ipv4> ();
  NS_ASSERT_MSG (ipv4, "DhcpHelper: NetDevice is associated"
                 " with a node without IPv4 stack installed -> fail "
                 "(maybe need to use InternetStackHelper?)");

  int32_t interface = ipv4->GetInterfaceForDevice (netDevice);
  if (interface == -1)
    {
      interface = ipv4->AddInterface (netDevice);
    }
  NS_ASSERT_MSG (interface >= 0, "DhcpHelper: Interface index not found");

  Ipv4InterfaceAddress ipv4Addr = Ipv4InterfaceAddress (serverAddr, poolMask);
  ipv4->AddAddress (interface, ipv4Addr);  // add ip address to that interface
  ipv4->SetMetric (interface, 1);          // setting cost of that interface
  ipv4->SetUp (interface);					// setting up the interface

  // Install the default traffic control configuration if the traffic
  // control layer has been aggregated, if this is not
  // a loopback interface, and there is no queue disc installed already
  Ptr<TrafficControlLayer> tc = node->GetObject<TrafficControlLayer> ();
  if (tc && DynamicCast<LoopbackNetDevice> (netDevice) == 0 && tc->GetRootQueueDiscOnDevice (netDevice) == 0)
    {
      NS_LOG_LOGIC ("DhcpHelper - Installing default traffic control configuration");
      TrafficControlHelper tcHelper = TrafficControlHelper::Default ();
      tcHelper.Install (netDevice);
    }

  // check that the already fixed addresses are not in conflict with the pool
  std::list<Ipv4Address>::iterator iter;
  for (iter=m_fixedAddresses.begin (); iter!=m_fixedAddresses.end (); iter ++)
    {
      if (iter->Get () >= minAddr.Get () && iter->Get () <= maxAddr.Get ())
        {
          NS_ABORT_MSG ("DhcpHelper: Fixed address can not conflict with a pool: " << *iter << " is in [" << minAddr << ",  " << maxAddr << "]");
        }
    }
  m_addressPools.push_back (std::make_pair (minAddr, maxAddr));  // m_addressPools - list address pools 

  Ptr<Application> app = m_serverFactory.Create<DhcpServer> ();
  node->AddApplication (app);      // invoking the DhcpServer Application
  return ApplicationContainer (app);   
}

// relay acting as a client

/*ApplicationContainer DhcpHelper::InstallDhcpRelay (Ptr<NetDevice> netDevice, Ipv4Address relayAddr,
                                                     Ipv4Mask subMask, Ipv4Address dhcps)
{
  m_relayFactory.Set ("RelayAddress", Ipv4AddressValue (relayAddr));
  m_relayFactory.Set ("SubnetMask", Ipv4MaskValue (subMask)); 
  m_relayFactory.Set ("DhcpServerAddress", Ipv4AddressValue (dhcps));
  
  Ptr<Node> node = netDevice->GetNode ();
  NS_ASSERT_MSG (node != 0, "DhcpHelper: NetDevice is not not associated with any node -> fail");

  Ptr<Ipv4> ipv4 = node->GetObject<Ipv4> ();
  NS_ASSERT_MSG (ipv4, "DhcpHelper: NetDevice is associated"
                 " with a node without IPv4 stack installed -> fail "
                 "(maybe need to use InternetStackHelper?)");

  int32_t interface = ipv4->GetInterfaceForDevice (netDevice);
  if (interface == -1)
    {
      interface = ipv4->AddInterface (netDevice);
    }
  NS_ASSERT_MSG (interface >= 0, "DhcpHelper: Interface index not found");

  Ipv4InterfaceAddress ipv4Addr = Ipv4InterfaceAddress (relayAddr, subMask);
  ipv4->AddAddress (interface, ipv4Addr);  
  ipv4->SetMetric (interface, 1);          
  ipv4->SetUp (interface);          

  Ptr<TrafficControlLayer> tc = node->GetObject<TrafficControlLayer> ();
  if (tc && DynamicCast<LoopbackNetDevice> (netDevice) == 0 && tc->GetRootQueueDiscOnDevice (netDevice) == 0)
    {
      NS_LOG_LOGIC ("DhcpHelper - Installing default traffic control configuration");
      TrafficControlHelper tcHelper = TrafficControlHelper::Default ();
      tcHelper.Install (netDevice);
    }  

  Ptr<Application> app = m_relayFactory.Create<DhcpRelay> ();
  node->AddApplication (app);      
  return ApplicationContainer (app);   
}



*/
 ApplicationContainer InstallDhcpRelay (NetDeviceContainer netDevices,Ipv4Address relayAddress,Ipv4Address relayAddressClientSide,
                                                            Ipv4Address dhcps,Ipv4Mask poolMask) 
 {

  ApplicationContainer apps;
  for (NetDeviceContainer::Iterator i = netDevices.Begin (); i != netDevices.End (); ++i) // inst
    {
        if(i==netDevices.Begin())
          apps.Add (InstallDhcpRelayClientSide (*i,relayAddressClientSide,poolMask));
        else
          apps.Add (InstallDhcpRelayServerSide (*i,relayAddress,dhcps,poolMask));

    }
  return apps;

 }
 Ptr<Application> InstallDhcpRelayClientSide(Ptr<NetDevice> netDevice,Ipv4Address relayAddressClientSide,Ipv4Mask mask)
 {
  
 }
 Ptr<Application> InstallDhcpRelayServerSide(Ptr<NetDevice> netDevice,Ipv4Address relayAddress,Ipv4Address dhcps,Ipv4Mask mask)
 {  m_relayFactory.Set ("RelayAddress", Ipv4AddressValue (relayAddress));
    m_relayFactory.Set ("SubnetMask", Ipv4MaskValue (mask)); 
    m_relayFactory.Set ("DhcpServerAddress", Ipv4AddressValue (dhcps));

    Ptr<Node> node = netDevice->GetNode ();
  NS_ASSERT_MSG (node != 0, "DhcpHelper: NetDevice is not not associated with any node -> fail");

  Ptr<Ipv4> ipv4 = node->GetObject<Ipv4> ();
  NS_ASSERT_MSG (ipv4, "DhcpHelper: NetDevice is associated"
                 " with a node without IPv4 stack installed -> fail "
                 "(maybe need to use InternetStackHelper?)");

  int32_t interface = ipv4->GetInterfaceForDevice (netDevice);
  if (interface == -1)
    {
      interface = ipv4->AddInterface (netDevice);
    }
  NS_ASSERT_MSG (interface >= 0, "DhcpHelper: Interface index not found");

  Ipv4InterfaceAddress ipv4Addr = Ipv4InterfaceAddress (relayAddress, mask);
  ipv4->AddAddress (interface, ipv4Addr);  
  ipv4->SetMetric (interface, 1);          
  ipv4->SetUp (interface); 

   Ptr<TrafficControlLayer> tc = node->GetObject<TrafficControlLayer> ();
  if (tc && DynamicCast<LoopbackNetDevice> (netDevice) == 0 && tc->GetRootQueueDiscOnDevice (netDevice) == 0)
    {
      NS_LOG_LOGIC ("DhcpHelper - Installing default traffic control configuration");
      TrafficControlHelper tcHelper = TrafficControlHelper::Default ();  
      tcHelper.Install (netDevice);   
    }

  Ptr<DhcpRelay> app = DynamicCast <DhcpRelay> (m_relayFactory.Create<DhcpRelay> ()); // create dhcpClient object
  app->SetDhcpClientNetDevice (netDevice);  // invoking a fnction of dhcpClient
  node->AddApplication (app);    

 }

  


Ipv4InterfaceContainer DhcpHelper::InstallFixedAddress (Ptr<NetDevice> netDevice, Ipv4Address addr, Ipv4Mask mask)
{
  Ipv4InterfaceContainer retval;

  Ptr<Node> node = netDevice->GetNode ();
  NS_ASSERT_MSG (node != 0, "DhcpHelper: NetDevice is not not associated with any node -> fail");

  Ptr<Ipv4> ipv4 = node->GetObject<Ipv4> ();
  NS_ASSERT_MSG (ipv4, "DhcpHelper: NetDevice is associated"
                 " with a node without IPv4 stack installed -> fail "
                 "(maybe need to use InternetStackHelper?)");

  int32_t interface = ipv4->GetInterfaceForDevice (netDevice);
  if (interface == -1)
    {
      interface = ipv4->AddInterface (netDevice);
    }
  NS_ASSERT_MSG (interface >= 0, "DhcpHelper: Interface index not found");

  Ipv4InterfaceAddress ipv4Addr = Ipv4InterfaceAddress (addr, mask);
  ipv4->AddAddress (interface, ipv4Addr);
  ipv4->SetMetric (interface, 1); 
  ipv4->SetUp (interface);
  retval.Add (ipv4, interface);

  // Install the default traffic control configuration if the traffic
  // control layer has been aggregated, if this is not
  // a loopback interface, and there is no queue disc installed already
  Ptr<TrafficControlLayer> tc = node->GetObject<TrafficControlLayer> ();
  if (tc && DynamicCast<LoopbackNetDevice> (netDevice) == 0 && tc->GetRootQueueDiscOnDevice (netDevice) == 0)
    {
      NS_LOG_LOGIC ("DhcpHelper - Installing default traffic control configuration");
      TrafficControlHelper tcHelper = TrafficControlHelper::Default ();
      tcHelper.Install (netDevice);
    }

  // check that the already fixed addresses are not in conflict with the pool
  std::list<std::pair<Ipv4Address, Ipv4Address> >::iterator iter;
  for (iter=m_addressPools.begin (); iter!=m_addressPools.end (); iter ++)
    {
      if (addr.Get () >= iter->first.Get () && addr.Get () <= iter->second.Get ())
        {
          NS_ABORT_MSG ("DhcpHelper: Fixed address can not conflict with a pool: " << addr << " is in [" << iter->first << ",  " << iter->second << "]");
        }
    }
  m_fixedAddresses.push_back (addr);
  return retval;
}

} // namespace ns3
